from breezy.errors import BzrError, DependencyNotPresent
from breezy.branch import Branch
def _update_paths(checker_manager, temp_prefix):
    temp_prefix_length = len(temp_prefix)
    for checker in checker_manager.checkers:
        filename = checker.display_name
        if filename.startswith(temp_prefix):
            checker.display_name = os.path.relpath(filename[temp_prefix_length:])