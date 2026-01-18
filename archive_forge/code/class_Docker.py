from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class Docker(base.Group):
    """Manage Artifact Registry container images and tags.

  To list images under repository `my-repo`, project `my-project`, in
  `us-central1`:

      $ {command} images list us-central1-docker.pkg.dev/my-project/my-repo

  To delete image `busy-box` in `us-west1` and all of its digests and tags:

      $ {command} images delete
      us-west1-docker.pkg.dev/my-project/my-repository/busy-box

  To add tag `my-tag` to image `busy-box` referenced by digest `abcxyz` in
  `us-west1`:

      $ {command} tags add
      us-west1-docker.pkg.dev/my-project/my-repository/busy-box@sha256:abcxyz
      us-west1-docker.pkg.dev/my-project/my-repository/busy-box:my-tag

  To delete tag `my-tag` from image `busy-box` in `us-west1`:

      $ {command} tags delete
      us-west1-docker.pkg.dev/my-project/my-repository/busy-box:my-tag

  To list tags for image `busy-box` in `us-west1`:

      $ {command} tags list
      us-west1-docker.pkg.dev/my-project/my-repository/busy-box
  """
    category = base.CI_CD_CATEGORY