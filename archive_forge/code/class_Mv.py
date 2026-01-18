from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.storage import cp_command_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.command_lib.storage import storage_url
class Mv(base.Command):
    """Moves or renames objects."""
    detailed_help = {'DESCRIPTION': '\n      The mv command allows you to move data between your local file system and\n      the cloud, move data within the cloud, and move data between cloud storage\n      providers.\n\n      *Renaming Groups Of Objects*\n\n      You can use the mv command to rename all objects with a given prefix to\n      have a new prefix. For example, the following command renames all objects\n      under gs://my_bucket/oldprefix to be under gs://my_bucket/newprefix,\n      otherwise preserving the naming structure:\n\n        $ {command} gs://my_bucket/oldprefix gs://my_bucket/newprefix\n\n      Note that when using mv to rename groups of objects with a common prefix,\n      you cannot specify the source URL using wildcards; you must spell out the\n      complete name.\n\n      If you do a rename as specified above and you want to preserve ACLs.\n\n      *Non-Atomic Operation*\n\n      Unlike the case with many file systems, the mv command does not perform a\n      single atomic operation. Rather, it performs a copy from source to\n      destination followed by removing the source for each object.\n\n      A consequence of this is that, in addition to normal network and operation\n      charges, if you move a Nearline Storage, Coldline Storage, or Archive\n      Storage object, deletion and data retrieval charges apply.\n      See the documentation for pricing details.\n      ', 'EXAMPLES': '\n\n      To move all objects from a bucket to a local directory you could use:\n\n        $ {command} gs://my_bucket/* dir\n\n      Similarly, to move all objects from a local directory to a bucket you\n      could use:\n\n        $ {command} ./dir gs://my_bucket\n\n      The following command renames all objects under gs://my_bucket/oldprefix\n      to be under gs://my_bucket/newprefix, otherwise preserving the naming\n      structure:\n\n        $ {command} gs://my_bucket/oldprefix gs://my_bucket/newprefix\n      '}

    @classmethod
    def Args(cls, parser):
        cp_command_util.add_cp_and_mv_flags(parser)
        flags.add_per_object_retention_flags(parser)

    def Run(self, args):
        for url_string in args.source:
            url = storage_url.storage_url_from_string(url_string)
            if isinstance(url, storage_url.CloudUrl) and (not url.is_object()):
                raise errors.InvalidUrlError('Cannot mv buckets.')
            if url.is_stdio:
                raise errors.InvalidUrlError('Cannot mv stdin.')
        args.recursive = True
        self.exit_code = cp_command_util.run_cp(args, delete_source=True)