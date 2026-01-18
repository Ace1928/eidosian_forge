import os
from parso import file_io
class FolderIO(AbstractFolderIO):

    def get_base_name(self):
        return os.path.basename(self.path)

    def list(self):
        return os.listdir(self.path)

    def get_file_io(self, name):
        return FileIO(os.path.join(self.path, name))

    def get_parent_folder(self):
        return FolderIO(os.path.dirname(self.path))

    def walk(self):
        for root, dirs, files in os.walk(self.path):
            root_folder_io = FolderIO(root)
            original_folder_ios = [FolderIO(os.path.join(root, d)) for d in dirs]
            modified_folder_ios = list(original_folder_ios)
            yield (root_folder_io, modified_folder_ios, [FileIO(os.path.join(root, f)) for f in files])
            modified_iterator = iter(reversed(modified_folder_ios))
            current = next(modified_iterator, None)
            i = len(original_folder_ios)
            for folder_io in reversed(original_folder_ios):
                i -= 1
                if current is folder_io:
                    current = next(modified_iterator, None)
                else:
                    del dirs[i]