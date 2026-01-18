from breezy.errors import BzrError, DependencyNotPresent
from breezy.branch import Branch
def _copy_files_to(tree, target_dir, files):
    from breezy import osutils
    for relpath in files:
        target_path = os.path.join(target_dir, relpath)
        os.makedirs(os.path.dirname(target_path))
        with tree.get_file(relpath) as inf, open(target_path, 'wb') as outf:
            osutils.pumpfile(inf, outf)
            yield target_path