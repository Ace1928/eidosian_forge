from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CommentControlValueValuesEnum(_messages.Enum):
    """If CommentControl is enabled, depending on the setting, builds may not
    fire until a repository writer comments `/gcbrun` on a pull request or
    `/gcbrun` is in the pull request description. Only PR comments that
    contain `/gcbrun` will trigger builds. If CommentControl is set to
    disabled, comments with `/gcbrun` from a user with repository write
    permission or above will still trigger builds to run.

    Values:
      COMMENTS_DISABLED: Do not require `/gcbrun` comments from a user with
        repository write permission or above on pull requests before builds
        are triggered. Comments that contain `/gcbrun` will still fire builds
        so this should be thought of as comments not required.
      COMMENTS_ENABLED: Builds will only fire in response to pull requests if:
        1. The pull request author has repository write permission or above
        and `/gcbrun` is in the PR description. 2. A user with repository
        writer permissions or above comments `/gcbrun` on a pull request
        authored by any user.
      COMMENTS_ENABLED_FOR_EXTERNAL_CONTRIBUTORS_ONLY: Builds will only fire
        in response to pull requests if: 1. The pull request author is a
        repository writer or above. 2. If the author does not have write
        permissions, a user with write permissions or above must comment
        `/gcbrun` in order to fire a build.
    """
    COMMENTS_DISABLED = 0
    COMMENTS_ENABLED = 1
    COMMENTS_ENABLED_FOR_EXTERNAL_CONTRIBUTORS_ONLY = 2