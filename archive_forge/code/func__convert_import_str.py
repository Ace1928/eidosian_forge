from .errors import BzrError, InternalBzrError
def _convert_import_str(self, import_str):
    """This converts a import string into an import map.

        This only understands 'import foo, foo.bar, foo.bar.baz as bing'

        :param import_str: The import string to process
        """
    if not import_str.startswith('import '):
        raise ValueError('bad import string {!r}'.format(import_str))
    import_str = import_str[len('import '):]
    for path in import_str.split(','):
        path = path.strip()
        if not path:
            continue
        as_hunks = path.split(' as ')
        if len(as_hunks) == 2:
            name = as_hunks[1].strip()
            module_path = as_hunks[0].strip().split('.')
            if name in self.imports:
                raise ImportNameCollision(name)
            if not module_path[0]:
                raise ImportError(path)
            self.imports[name] = (module_path, None, {})
        else:
            module_path = path.split('.')
            name = module_path[0]
            if not name:
                raise ImportError(path)
            if name not in self.imports:
                module_def = ([name], None, {})
                self.imports[name] = module_def
            else:
                module_def = self.imports[name]
            cur_path = [name]
            cur = module_def[2]
            for child in module_path[1:]:
                cur_path.append(child)
                if child in cur:
                    cur = cur[child][2]
                else:
                    next = (cur_path[:], None, {})
                    cur[child] = next
                    cur = next[2]