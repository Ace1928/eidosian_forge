from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.storage import cp_command_util
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.command_lib.storage import rsync_command_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.tasks import get_sorted_list_file_task
from googlecloudsdk.command_lib.storage.tasks import task_executor
from googlecloudsdk.command_lib.storage.tasks import task_graph_executor
from googlecloudsdk.command_lib.storage.tasks import task_status
class Rsync(base.Command):
    """Synchronize content of two buckets/directories."""
    detailed_help = {'DESCRIPTION': '\n      *{command}* copies to and updates objects at `DESTINATION` to match\n      `SOURCE`. `SOURCE` must specify a directory, bucket, or bucket\n      subdirectory. *{command}* does not copy empty directory trees,\n      since storage providers use a [flat namespace](https://cloud.google.com/storage/docs/folders).\n\n      Note, shells (like bash, zsh) sometimes attempt to expand wildcards in\n      ways that can be surprising. Also, attempting to copy files whose names\n      contain wildcard characters can result in problems.\n\n      If synchronizing a large amount of data between clouds you might consider\n      setting up a Google Compute Engine account and running *{command}* there.\n      Since *{command}* cross-provider data transfers flow through the machine\n      where *{command}* is running, doing this can make your transfer run\n      significantly faster than on your local workstation.\n\n      ', 'EXAMPLES': '\n      To sync the contents of the local directory `data` to the bucket\n      gs://my-bucket/data:\n\n        $ {command} data gs://my-bucket/data\n\n      To recurse into directories use `--recursive`:\n\n        $ {command} data gs://my-bucket/data --recursive\n\n      To make the local directory `my-data` the same as the contents of\n      gs://mybucket/data and delete objects in the local directory that are\n      not in gs://mybucket/data:\n\n        $ {command} gs://mybucket/data my-data --recursive            --delete-unmatched-destination-objects\n\n      To make the contents of gs://mybucket2 the same as gs://mybucket1 and\n      delete objects in gs://mybucket2 that are not in gs://mybucket1:\n\n        $ {command} gs://mybucket1 gs://mybucket2 --recursive            --delete-unmatched-destination-objects\n\n      To copy all objects from `dir1` into `dir2` and delete all objects\n      in `dir2` which are not in `dir1`:\n\n        $ {command} dir1 dir2 --recursive -           --delete-unmatched-destination-objects\n\n      To mirror your objects across cloud providers:\n\n        $ {command} gs://my-gs-bucket s3://my-s3-bucket --recursive            --delete-unmatched-destination-objects\n\n      To apply gzip compression to only uploaded image files in `dir`:\n\n        $ {command} dir gs://my-bucket/data --gzip-in-flight=jpeg,jpg,gif,png\n\n      To skip the file `dir/data1/a.txt`:\n\n        $ {command} dir gs://my-bucket --exclude="data./.*\\.txt$"\n\n      To skip all .txt and .jpg files:\n\n        $ {command} dir gs://my-bucket --exclude=".*\\.txt$|.*\\.jpg$"\n      '}

    @classmethod
    def Args(cls, parser):
        parser.add_argument('source', help='The source container path.')
        parser.add_argument('destination', help='The destination container path.')
        acl_flags_group = parser.add_group()
        flags.add_preserve_acl_flag(acl_flags_group, hidden=True)
        flags.add_predefined_acl_flag(acl_flags_group)
        flags.add_encryption_flags(parser)
        cp_command_util.add_cp_mv_rsync_flags(parser)
        cp_command_util.add_gzip_in_flight_flags(parser)
        cp_command_util.add_ignore_symlinks_flag(parser, default=True)
        cp_command_util.add_recursion_flag(parser)
        cp_command_util.add_include_managed_folders_flag(parser)
        parser.add_argument('--checksums-only', action='store_true', help='When comparing objects with matching names at the source and destination, skip modification time check and compare object hashes. Normally, hashes are only compared if modification times are not available.')
        parser.add_argument('--delete-unmatched-destination-objects', action='store_true', help=textwrap.dedent('            Delete extra files under DESTINATION not found under SOURCE.\n            By default extra files are not deleted. Managed folders are not\n            affected by this flag.\n\n            Note: this option can delete data quickly if you specify the wrong\n            source and destination combination.'))
        parser.add_argument('--dry-run', action='store_true', help='Print what operations rsync would perform without actually executing them.')
        parser.add_argument('-x', '--exclude', metavar='REGEX', type=arg_parsers.ArgList(), help='Exclude objects matching regex pattern from rsync.\n\nNote that this is a Python regular expression, not a pure wildcard\npattern. For example, matching a string ending in "abc" is\n`.*abc$` rather than `*abc`. Also note that the exclude path\nis relative, as opposed to absolute\n(similar to Linux `rsync` and `tar` exclude options).\n\nFor the Windows cmd.exe command line interpreter, use\n`^` as an escape character instead of `\\` and escape the `|`\ncharacter. When using Windows PowerShell, use `\'` instead of\n`"` and surround the `|` character with `"`.')
        parser.add_argument('-u', '--skip-if-dest-has-newer-mtime', action='store_true', help='Skip operating on destination object if it has a newer modification time than the source.')

    def Run(self, args):
        encryption_util.initialize_key_store(args)
        cp_command_util.validate_include_managed_folders(args, [args.source], storage_url.storage_url_from_string(args.destination))
        source_container = rsync_command_util.get_existing_container_resource(os.path.expanduser(args.source), args.ignore_symlinks)
        destination_container = rsync_command_util.get_existing_or_placeholder_destination_resource(os.path.expanduser(args.destination), args.ignore_symlinks)
        list_tasks, cleanup_paths = _get_list_tasks_and_cleanup_paths(args, source_container, destination_container)
        try:
            exit_code = task_executor.execute_tasks(list_tasks, continue_on_error=args.continue_on_error, parallelizable=True)
            if exit_code:
                self.exit_code = exit_code
                return
            if args.include_managed_folders:
                exit_code = _perform_rsync(args, source_container, destination_container, perform_managed_folder_operations=True)
                if exit_code:
                    self.exit_code = exit_code
                    return
            self.exit_code = _perform_rsync(args, source_container, destination_container, perform_managed_folder_operations=False)
        finally:
            for path in cleanup_paths:
                rsync_command_util.try_to_delete_file(path)