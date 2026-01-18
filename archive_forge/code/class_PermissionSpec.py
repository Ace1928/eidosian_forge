import os
class PermissionSpec(object):
    """
    Represents a set of specifications for permissions.

    Typically reads from a file that looks like this::

      rwxrwxrwx user:group filename

    If the filename ends in /, then it expected to be a directory, and
    the directory is made executable automatically, and the contents
    of the directory are given the same permission (recursively).  By
    default the executable bit on files is left as-is, unless the
    permissions specifically say it should be on in some way.

    You can use 'nomodify filename' for permissions to say that any
    permission is okay, and permissions should not be changed.

    Use 'noexist filename' to say that a specific file should not
    exist.

    Use 'symlink filename symlinked_to' to assert a symlink destination

    The entire file is read, and most specific rules are used for each
    file (i.e., a rule for a subdirectory overrides the rule for a
    superdirectory).  Order does not matter.
    """

    def __init__(self):
        self.paths = {}

    def parsefile(self, filename):
        f = open(filename)
        lines = f.readlines()
        f.close()
        self.parselines(lines, filename=filename)
    commands = {}

    def parselines(self, lines, filename=None):
        for lineindex, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            command = parts[0]
            if command in self.commands:
                cmd = self.commands[command](*parts[1:])
            else:
                cmd = self.commands['*'](*parts)
            self.paths[cmd.path] = cmd

    def check(self):
        action = _Check(self)
        self.traverse(action)

    def fix(self):
        action = _Fixer(self)
        self.traverse(action)

    def traverse(self, action):
        paths = self.paths_sorted()
        checked = {}
        for path, checker in list(paths)[::-1]:
            self.check_tree(action, path, paths, checked)
        for path, checker in paths:
            if path not in checked:
                action.noexists(path, checker)

    def traverse_tree(self, action, path, paths, checked):
        if path in checked:
            return
        self.traverse_path(action, path, paths, checked)
        if os.path.isdir(path):
            for fn in os.listdir(path):
                fn = os.path.join(path, fn)
                self.traverse_tree(action, fn, paths, checked)

    def traverse_path(self, action, path, paths, checked):
        checked[path] = None
        for check_path, checker in paths:
            if path.startswith(check_path):
                action.check(check_path, checker)
                if not checker.inherit:
                    break

    def paths_sorted(self):
        paths = sorted(self.paths.items(), key=lambda key_value: len(key_value[0]), reversed=True)