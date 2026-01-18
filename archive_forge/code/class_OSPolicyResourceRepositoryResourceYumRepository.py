from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyResourceRepositoryResourceYumRepository(_messages.Message):
    """Represents a single yum package repository. These are added to a repo
  file that is managed at `/etc/yum.repos.d/google_osconfig.repo`.

  Fields:
    baseUrl: Required. The location of the repository directory.
    displayName: The display name of the repository.
    gpgKeys: URIs of GPG keys.
    id: Required. A one word, unique name for this repository. This is the
      `repo id` in the yum config file and also the `display_name` if
      `display_name` is omitted. This id is also used as the unique identifier
      when checking for resource conflicts.
  """
    baseUrl = _messages.StringField(1)
    displayName = _messages.StringField(2)
    gpgKeys = _messages.StringField(3, repeated=True)
    id = _messages.StringField(4)