import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
class _PastaEditVisitor(ast.NodeVisitor):
    """AST Visitor that processes function calls.

  Updates function calls from old API version to new API version using a given
  change spec.
  """

    def __init__(self, api_change_spec):
        self._api_change_spec = api_change_spec
        self._log = []
        self._stack = []

    def visit(self, node):
        self._stack.append(node)
        super(_PastaEditVisitor, self).visit(node)
        self._stack.pop()

    @property
    def errors(self):
        return [log for log in self._log if log[0] == ERROR]

    @property
    def warnings(self):
        return [log for log in self._log if log[0] == WARNING]

    @property
    def warnings_and_errors(self):
        return [log for log in self._log if log[0] in (WARNING, ERROR)]

    @property
    def info(self):
        return [log for log in self._log if log[0] == INFO]

    @property
    def log(self):
        return self._log

    def add_log(self, severity, lineno, col, msg):
        self._log.append((severity, lineno, col, msg))
        print('%s line %d:%d: %s' % (severity, lineno, col, msg))

    def add_logs(self, logs):
        """Record a log and print it.

    The log should be a tuple `(severity, lineno, col_offset, msg)`, which will
    be printed and recorded. It is part of the log available in the `self.log`
    property.

    Args:
      logs: The logs to add. Must be a list of tuples
        `(severity, lineno, col_offset, msg)`.
    """
        self._log.extend(logs)
        for log in logs:
            print('%s line %d:%d: %s' % log)

    def _get_applicable_entries(self, transformer_field, full_name, name):
        """Get all list entries indexed by name that apply to full_name or name."""
        function_transformers = getattr(self._api_change_spec, transformer_field, {})
        glob_name = '*.' + name if name else None
        transformers = []
        if full_name in function_transformers:
            transformers.append(function_transformers[full_name])
        if glob_name in function_transformers:
            transformers.append(function_transformers[glob_name])
        if '*' in function_transformers:
            transformers.append(function_transformers['*'])
        return transformers

    def _get_applicable_dict(self, transformer_field, full_name, name):
        """Get all dict entries indexed by name that apply to full_name or name."""
        function_transformers = getattr(self._api_change_spec, transformer_field, {})
        glob_name = '*.' + name if name else None
        transformers = function_transformers.get('*', {}).copy()
        transformers.update(function_transformers.get(glob_name, {}))
        transformers.update(function_transformers.get(full_name, {}))
        return transformers

    def _get_full_name(self, node):
        """Traverse an Attribute node to generate a full name, e.g., "tf.foo.bar".

    This is the inverse of `full_name_node`.

    Args:
      node: A Node of type Attribute.

    Returns:
      a '.'-delimited full-name or None if node was not Attribute or Name.
      i.e. `foo()+b).bar` returns None, while `a.b.c` would return "a.b.c".
    """
        curr = node
        items = []
        while not isinstance(curr, ast.Name):
            if not isinstance(curr, ast.Attribute):
                return None
            items.append(curr.attr)
            curr = curr.value
        items.append(curr.id)
        return '.'.join(reversed(items))

    def _maybe_add_warning(self, node, full_name):
        """Adds an error to be printed about full_name at node."""
        function_warnings = self._api_change_spec.function_warnings
        if full_name in function_warnings:
            level, message = function_warnings[full_name]
            message = message.replace('<function name>', full_name)
            self.add_log(level, node.lineno, node.col_offset, '%s requires manual check. %s' % (full_name, message))
            return True
        else:
            return False

    def _maybe_add_module_deprecation_warning(self, node, full_name, whole_name):
        """Adds a warning if full_name is a deprecated module."""
        warnings = self._api_change_spec.module_deprecations
        if full_name in warnings:
            level, message = warnings[full_name]
            message = message.replace('<function name>', whole_name)
            self.add_log(level, node.lineno, node.col_offset, 'Using member %s in deprecated module %s. %s' % (whole_name, full_name, message))
            return True
        else:
            return False

    def _maybe_add_call_warning(self, node, full_name, name):
        """Print a warning when specific functions are called with selected args.

    The function _print_warning_for_function matches the full name of the called
    function, e.g., tf.foo.bar(). This function matches the function name that
    is called, as long as the function is an attribute. For example,
    `tf.foo.bar()` and `foo.bar()` are matched, but not `bar()`.

    Args:
      node: ast.Call object
      full_name: The precomputed full name of the callable, if one exists, None
        otherwise.
      name: The precomputed name of the callable, if one exists, None otherwise.

    Returns:
      Whether an error was recorded.
    """
        warned = False
        if isinstance(node.func, ast.Attribute):
            warned = self._maybe_add_warning(node, '*.' + name)
        arg_warnings = self._get_applicable_dict('function_arg_warnings', full_name, name)
        variadic_args = uses_star_args_or_kwargs_in_call(node)
        for (kwarg, arg), (level, warning) in sorted(arg_warnings.items()):
            present, _ = get_arg_value(node, kwarg, arg) or variadic_args
            if present:
                warned = True
                warning_message = warning.replace('<function name>', full_name or name)
                template = '%s called with %s argument, requires manual check: %s'
                if variadic_args:
                    template = '%s called with *args or **kwargs that may include %s, requires manual check: %s'
                self.add_log(level, node.lineno, node.col_offset, template % (full_name or name, kwarg, warning_message))
        return warned

    def _maybe_rename(self, parent, node, full_name):
        """Replace node (Attribute or Name) with a node representing full_name."""
        new_name = self._api_change_spec.symbol_renames.get(full_name, None)
        if new_name:
            self.add_log(INFO, node.lineno, node.col_offset, 'Renamed %r to %r' % (full_name, new_name))
            new_node = full_name_node(new_name, node.ctx)
            ast.copy_location(new_node, node)
            pasta.ast_utils.replace_child(parent, node, new_node)
            return True
        else:
            return False

    def _maybe_change_to_function_call(self, parent, node, full_name):
        """Wraps node (typically, an Attribute or Expr) in a Call."""
        if full_name in self._api_change_spec.change_to_function:
            if not isinstance(parent, ast.Call):
                new_node = ast.Call(node, [], [])
                pasta.ast_utils.replace_child(parent, node, new_node)
                ast.copy_location(new_node, node)
                self.add_log(INFO, node.lineno, node.col_offset, 'Changed %r to a function call' % full_name)
                return True
        return False

    def _maybe_add_arg_names(self, node, full_name):
        """Make args into keyword args if function called full_name requires it."""
        function_reorders = self._api_change_spec.function_reorders
        if full_name in function_reorders:
            if uses_star_args_in_call(node):
                self.add_log(WARNING, node.lineno, node.col_offset, '(Manual check required) upgrading %s may require re-ordering the call arguments, but it was passed variable-length positional *args. The upgrade script cannot handle these automatically.' % full_name)
            reordered = function_reorders[full_name]
            new_args = []
            new_keywords = []
            idx = 0
            for arg in node.args:
                if sys.version_info[:2] >= (3, 5) and isinstance(arg, ast.Starred):
                    continue
                keyword_arg = reordered[idx]
                if keyword_arg:
                    new_keywords.append(ast.keyword(arg=keyword_arg, value=arg))
                else:
                    new_args.append(arg)
                idx += 1
            if new_keywords:
                self.add_log(INFO, node.lineno, node.col_offset, 'Added keywords to args of function %r' % full_name)
                node.args = new_args
                node.keywords = new_keywords + (node.keywords or [])
                return True
        return False

    def _maybe_modify_args(self, node, full_name, name):
        """Rename keyword args if the function called full_name requires it."""
        renamed_keywords = self._get_applicable_dict('function_keyword_renames', full_name, name)
        if not renamed_keywords:
            return False
        if uses_star_kwargs_in_call(node):
            self.add_log(WARNING, node.lineno, node.col_offset, '(Manual check required) upgrading %s may require renaming or removing call arguments, but it was passed variable-length *args or **kwargs. The upgrade script cannot handle these automatically.' % (full_name or name))
        modified = False
        new_keywords = []
        for keyword in node.keywords:
            argkey = keyword.arg
            if argkey in renamed_keywords:
                modified = True
                if renamed_keywords[argkey] is None:
                    lineno = getattr(keyword, 'lineno', node.lineno)
                    col_offset = getattr(keyword, 'col_offset', node.col_offset)
                    self.add_log(INFO, lineno, col_offset, 'Removed argument %s for function %s' % (argkey, full_name or name))
                else:
                    keyword.arg = renamed_keywords[argkey]
                    lineno = getattr(keyword, 'lineno', node.lineno)
                    col_offset = getattr(keyword, 'col_offset', node.col_offset)
                    self.add_log(INFO, lineno, col_offset, 'Renamed keyword argument for %s from %s to %s' % (full_name, argkey, renamed_keywords[argkey]))
                    new_keywords.append(keyword)
            else:
                new_keywords.append(keyword)
        if modified:
            node.keywords = new_keywords
        return modified

    def visit_Call(self, node):
        """Handle visiting a call node in the AST.

    Args:
      node: Current Node
    """
        assert self._stack[-1] is node
        full_name = self._get_full_name(node.func)
        if full_name:
            name = full_name.split('.')[-1]
        elif isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            name = None
        self._maybe_add_call_warning(node, full_name, name)
        self._maybe_add_arg_names(node, full_name)
        self._maybe_modify_args(node, full_name, name)
        transformers = self._get_applicable_entries('function_transformers', full_name, name)
        parent = self._stack[-2]
        if transformers:
            if uses_star_args_or_kwargs_in_call(node):
                self.add_log(WARNING, node.lineno, node.col_offset, '(Manual check required) upgrading %s may require modifying call arguments, but it was passed variable-length *args or **kwargs. The upgrade script cannot handle these automatically.' % (full_name or name))
        for transformer in transformers:
            logs = []
            new_node = transformer(parent, node, full_name, name, logs)
            self.add_logs(logs)
            if new_node and new_node is not node:
                pasta.ast_utils.replace_child(parent, node, new_node)
                node = new_node
                self._stack[-1] = node
        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Handle bare Attributes i.e. [tf.foo, tf.bar]."""
        assert self._stack[-1] is node
        full_name = self._get_full_name(node)
        if full_name:
            parent = self._stack[-2]
            self._maybe_add_warning(node, full_name)
            if self._maybe_rename(parent, node, full_name):
                return
            if self._maybe_change_to_function_call(parent, node, full_name):
                return
            i = 2
            while isinstance(self._stack[-i], ast.Attribute):
                i += 1
            whole_name = pasta.dump(self._stack[-(i - 1)])
            self._maybe_add_module_deprecation_warning(node, full_name, whole_name)
        self.generic_visit(node)

    def visit_Import(self, node):
        """Handle visiting an import node in the AST.

    Args:
      node: Current Node
    """
        new_aliases = []
        import_updated = False
        import_renames = getattr(self._api_change_spec, 'import_renames', {})
        max_submodule_depth = getattr(self._api_change_spec, 'max_submodule_depth', 1)
        inserts_after_imports = getattr(self._api_change_spec, 'inserts_after_imports', {})
        for import_alias in node.names:
            all_import_components = import_alias.name.split('.')
            found_update = False
            for i in reversed(list(range(1, max_submodule_depth + 1))):
                import_component = all_import_components[0]
                for j in range(1, min(i, len(all_import_components))):
                    import_component += '.' + all_import_components[j]
                import_rename_spec = import_renames.get(import_component, None)
                if not import_rename_spec or excluded_from_module_rename(import_alias.name, import_rename_spec):
                    continue
                new_name = import_rename_spec.new_name + import_alias.name[len(import_component):]
                new_asname = import_alias.asname
                if not new_asname and '.' not in import_alias.name:
                    new_asname = import_alias.name
                new_alias = ast.alias(name=new_name, asname=new_asname)
                new_aliases.append(new_alias)
                import_updated = True
                found_update = True
                full_import = (import_alias.name, import_alias.asname)
                insert_offset = 1
                for line_to_insert in inserts_after_imports.get(full_import, []):
                    assert self._stack[-1] is node
                    parent = self._stack[-2]
                    new_line_node = pasta.parse(line_to_insert)
                    ast.copy_location(new_line_node, node)
                    parent.body.insert(parent.body.index(node) + insert_offset, new_line_node)
                    insert_offset += 1
                    old_suffix = pasta.base.formatting.get(node, 'suffix')
                    if old_suffix is None:
                        old_suffix = os.linesep
                    if os.linesep not in old_suffix:
                        pasta.base.formatting.set(node, 'suffix', old_suffix + os.linesep)
                    pasta.base.formatting.set(new_line_node, 'prefix', pasta.base.formatting.get(node, 'prefix'))
                    pasta.base.formatting.set(new_line_node, 'suffix', os.linesep)
                    self.add_log(INFO, node.lineno, node.col_offset, 'Adding `%s` after import of %s' % (new_line_node, import_alias.name))
                if found_update:
                    break
            if not found_update:
                new_aliases.append(import_alias)
        if import_updated:
            assert self._stack[-1] is node
            parent = self._stack[-2]
            new_node = ast.Import(new_aliases)
            ast.copy_location(new_node, node)
            pasta.ast_utils.replace_child(parent, node, new_node)
            self.add_log(INFO, node.lineno, node.col_offset, 'Changed import from %r to %r.' % (pasta.dump(node), pasta.dump(new_node)))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Handle visiting an import-from node in the AST.

    Args:
      node: Current Node
    """
        if not node.module:
            self.generic_visit(node)
            return
        from_import = node.module
        from_import_first_component = from_import.split('.')[0]
        import_renames = getattr(self._api_change_spec, 'import_renames', {})
        import_rename_spec = import_renames.get(from_import_first_component, None)
        if not import_rename_spec:
            self.generic_visit(node)
            return
        updated_aliases = []
        same_aliases = []
        for import_alias in node.names:
            full_module_name = '%s.%s' % (from_import, import_alias.name)
            if excluded_from_module_rename(full_module_name, import_rename_spec):
                same_aliases.append(import_alias)
            else:
                updated_aliases.append(import_alias)
        if not updated_aliases:
            self.generic_visit(node)
            return
        assert self._stack[-1] is node
        parent = self._stack[-2]
        new_from_import = import_rename_spec.new_name + from_import[len(from_import_first_component):]
        updated_node = ast.ImportFrom(new_from_import, updated_aliases, node.level)
        ast.copy_location(updated_node, node)
        pasta.ast_utils.replace_child(parent, node, updated_node)
        additional_import_log = ''
        if same_aliases:
            same_node = ast.ImportFrom(from_import, same_aliases, node.level, col_offset=node.col_offset, lineno=node.lineno)
            ast.copy_location(same_node, node)
            parent.body.insert(parent.body.index(updated_node), same_node)
            pasta.base.formatting.set(same_node, 'prefix', pasta.base.formatting.get(updated_node, 'prefix'))
            additional_import_log = ' and %r' % pasta.dump(same_node)
        self.add_log(INFO, node.lineno, node.col_offset, 'Changed import from %r to %r%s.' % (pasta.dump(node), pasta.dump(updated_node), additional_import_log))
        self.generic_visit(node)