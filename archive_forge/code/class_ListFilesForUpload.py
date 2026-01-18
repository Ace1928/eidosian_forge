from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import gcloudignore
class ListFilesForUpload(base.Command):
    """List files for upload.

  List the files that would be uploaded in a given directory.

  Useful for checking the effects of a .gitignore or .gcloudignore file.
  """

    @staticmethod
    def Args(parser):
        parser.add_argument('directory', default='.', nargs='?', help='The directory in which to show what files would be uploaded')
        parser.display_info.AddFormat('value(.)')

    def Run(self, args):
        file_chooser = gcloudignore.GetFileChooserForDir(args.directory, write_on_disk=False)
        file_chooser = file_chooser or gcloudignore.FileChooser([])
        return file_chooser.GetIncludedFiles(args.directory, include_dirs=False)