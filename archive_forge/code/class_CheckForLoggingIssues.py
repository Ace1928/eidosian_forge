import re
import ast
from hacking import core
class CheckForLoggingIssues(BaseASTChecker):
    DEBUG_CHECK_DESC = 'O324 Using translated string in debug logging'
    NONDEBUG_CHECK_DESC = 'O325 Not using translating helper for logging'
    EXCESS_HELPER_CHECK_DESC = 'O326 Using hints when _ is necessary'
    LOG_MODULES = 'logging'
    name = 'check_for_logging_issues'
    version = '1.0'

    def __init__(self, tree, filename):
        super(CheckForLoggingIssues, self).__init__(tree, filename)
        self.logger_names = []
        self.logger_module_names = []
        self.assignments = {}

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        item._parent = node
                        self.visit(item)
            elif isinstance(value, ast.AST):
                value._parent = node
                self.visit(value)

    def _filter_imports(self, module_name, alias):
        """Keeps lists of logging."""
        if module_name in self.LOG_MODULES:
            self.logger_module_names.append(alias.asname or alias.name)

    def visit_Import(self, node):
        for alias in node.names:
            self._filter_imports(alias.name, alias)
        return super(CheckForLoggingIssues, self).generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            full_name = '%s.%s' % (node.module, alias.name)
            self._filter_imports(full_name, alias)
        return super(CheckForLoggingIssues, self).generic_visit(node)

    def _find_name(self, node):
        """Return the fully qualified name or a Name or Attribute."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute) and isinstance(node.value, (ast.Name, ast.Attribute)):
            method_name = node.attr
            obj_name = self._find_name(node.value)
            if obj_name is None:
                return None
            return obj_name + '.' + method_name
        elif isinstance(node, str):
            return node
        else:
            return None

    def visit_Assign(self, node):
        """Look for 'LOG = logging.getLogger'

        This handles the simple case:
          name = [logging_module].getLogger(...)
        """
        attr_node_types = (ast.Name, ast.Attribute)
        if len(node.targets) != 1 or not isinstance(node.targets[0], attr_node_types):
            return super(CheckForLoggingIssues, self).generic_visit(node)
        target_name = self._find_name(node.targets[0])
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Mod):
            if isinstance(node.value.left, ast.Call) and isinstance(node.value.left.func, ast.Name):
                node = ast.Assign(value=node.value.left)
        if not isinstance(node.value, ast.Call):
            self.assignments.pop(target_name, None)
            return super(CheckForLoggingIssues, self).generic_visit(node)
        if isinstance(node.value.func, ast.Name):
            self.assignments[target_name] = node.value.func.id
            return super(CheckForLoggingIssues, self).generic_visit(node)
        if not isinstance(node.value.func, ast.Attribute) or not isinstance(node.value.func.value, attr_node_types):
            return super(CheckForLoggingIssues, self).generic_visit(node)
        object_name = self._find_name(node.value.func.value)
        func_name = node.value.func.attr
        if object_name in self.logger_module_names and func_name == 'getLogger':
            self.logger_names.append(target_name)
        return super(CheckForLoggingIssues, self).generic_visit(node)

    def visit_Call(self, node):
        """Look for the 'LOG.*' calls."""
        if isinstance(node.func, ast.Attribute):
            obj_name = self._find_name(node.func.value)
            if isinstance(node.func.value, ast.Name):
                method_name = node.func.attr
            elif isinstance(node.func.value, ast.Attribute):
                obj_name = self._find_name(node.func.value)
                method_name = node.func.attr
            else:
                return super(CheckForLoggingIssues, self).generic_visit(node)
            if obj_name in self.logger_names and method_name == 'warn':
                msg = node.args[0]
                self.add_error(msg, message=self.USING_DEPRECATED_WARN)
            if obj_name not in self.logger_names:
                return super(CheckForLoggingIssues, self).generic_visit(node)
            if not node.args:
                return super(CheckForLoggingIssues, self).generic_visit(node)
            if method_name == 'debug':
                self._process_debug(node)
        return super(CheckForLoggingIssues, self).generic_visit(node)

    def _process_debug(self, node):
        msg = node.args[0]
        if isinstance(msg, ast.Call) and isinstance(msg.func, ast.Name):
            self.add_error(msg, message=self.DEBUG_CHECK_DESC)
        elif isinstance(msg, ast.Name) and msg.id in self.assignments and (not self._is_raised_later(node, msg.id)):
            self.add_error(msg, message=self.DEBUG_CHECK_DESC)

    def _process_non_debug(self, node, method_name):
        msg = node.args[0]
        if isinstance(msg, ast.Call):
            self.add_error(msg, message=self.NONDEBUG_CHECK_DESC)
        elif isinstance(msg, ast.Name):
            if msg.id not in self.assignments:
                return
            if self._is_raised_later(node, msg.id):
                self.add_error(msg, message=self.NONDEBUG_CHECK_DESC)
            elif self._is_raised_later(node, msg.id):
                self.add_error(msg, message=self.EXCESS_HELPER_CHECK_DESC)

    def _is_raised_later(self, node, name):

        def find_peers(node):
            node_for_line = node._parent
            for _field, value in ast.iter_fields(node._parent._parent):
                if isinstance(value, list) and node_for_line in value:
                    return value[value.index(node_for_line) + 1:]
                continue
            return []
        peers = find_peers(node)
        for peer in peers:
            if isinstance(peer, ast.Raise):
                exc = peer.exc
                if isinstance(exc, ast.Call) and len(exc.args) > 0 and isinstance(exc.args[0], ast.Name) and (name in (a.id for a in exc.args)):
                    return True
                else:
                    return False
            elif isinstance(peer, ast.Assign):
                if name in (t.id for t in peer.targets if hasattr(t, 'id')):
                    return False