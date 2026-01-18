from io import BytesIO
from . import osutils, progress, trace
from .i18n import gettext
from .ui import ui_factory
def add_file_edge_hashes(self, tree, file_ids):
    """Update to reflect the hashes for files in the tree.

        :param tree: The tree containing the files.
        :param file_ids: A list of file_ids to perform the updates for.
        """
    desired_files = [(tree.id2path(f), f) for f in file_ids]
    with ui_factory.nested_progress_bar() as task:
        for num, (file_id, contents) in enumerate(tree.iter_files_bytes(desired_files)):
            task.update(gettext('Calculating hashes'), num, len(file_ids))
            s = BytesIO()
            s.writelines(contents)
            s.seek(0)
            self.add_edge_hashes(s.readlines(), file_id)