from io import BytesIO
from . import tree
from .filters import ContentFilterContext, filtered_output_bytes
class ContentFilterTree(tree.Tree):
    """A virtual tree that applies content filters to an underlying tree.

    Not every operation is supported yet.
    """

    def __init__(self, backing_tree, filter_stack_callback):
        """Construct a new filtered tree view.

        :param filter_stack_callback: A callable taking a path that returns
            the filter stack that should be used for that path.
        :param backing_tree: An underlying tree to wrap.
        """
        self.backing_tree = backing_tree
        self.filter_stack_callback = filter_stack_callback

    def get_file_text(self, path):
        chunks = self.backing_tree.get_file_lines(path)
        filters = self.filter_stack_callback(path)
        context = ContentFilterContext(path, self)
        contents = filtered_output_bytes(chunks, filters, context)
        content = b''.join(contents)
        return content

    def get_file(self, path):
        return BytesIO(self.get_file_text(path))

    def has_filename(self, filename):
        return self.backing_tree.has_filename

    def is_executable(self, path):
        return self.backing_tree.is_executable(path)

    def iter_entries_by_dir(self, specific_files=None, recurse_nested=False):
        return self.backing_tree.iter_entries_by_dir(specific_files=specific_files, recurse_nested=recurse_nested)

    def lock_read(self):
        return self.backing_tree.lock_read()

    def unlock(self):
        return self.backing_tree.unlock()