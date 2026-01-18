from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PostStartupScriptOptionValueValuesEnum(_messages.Enum):
    """Optional. Specifies the behavior of post startup script during
    migration.

    Values:
      POST_STARTUP_SCRIPT_OPTION_UNSPECIFIED: Post startup script option is
        not specified. Default is POST_STARTUP_SCRIPT_OPTION_SKIP.
      POST_STARTUP_SCRIPT_OPTION_SKIP: Not migrate the post startup script to
        the new Workbench Instance.
      POST_STARTUP_SCRIPT_OPTION_RERUN: Redownload and rerun the same post
        startup script as the Google-Managed Notebook.
    """
    POST_STARTUP_SCRIPT_OPTION_UNSPECIFIED = 0
    POST_STARTUP_SCRIPT_OPTION_SKIP = 1
    POST_STARTUP_SCRIPT_OPTION_RERUN = 2