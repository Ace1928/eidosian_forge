from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import semver
import six
class SDKDefinition(object):
    """Top level object for then entire component snapshot.

  Attributes:
    revision: int, The unique, monotonically increasing version of the snapshot.
    release_notes_url: string, The URL where the latest release notes can be
      downloaded.
    version: str, The version name of this release (i.e. 1.2.3).  This should be
      used only for informative purposes during an update (to say what version
      you are updating to).
    gcloud_rel_path: str, The path to the gcloud entrypoint relative to the SDK
      root.
    post_processing_command: str, The gcloud subcommand to run to do
      post-processing after an update.  This will be split on spaces before
      being executed.
    components: [Component], The component definitions.
    notifications: [NotificationSpec], The active update notifications.
  """

    @classmethod
    def FromDictionary(cls, dictionary):
        p = cls._ParseBase(dictionary)
        p.Parse('revision', required=True)
        p.Parse('release_notes_url')
        p.Parse('version')
        p.Parse('gcloud_rel_path')
        p.Parse('post_processing_command')
        p.ParseList('components', required=True, func=Component.FromDictionary)
        p.ParseList('notifications', default=[], func=NotificationSpec.FromDictionary)
        return cls(**p.Args())

    @classmethod
    def SchemaVersion(cls, dictionary):
        return cls._ParseBase(dictionary).Args()['schema_version']

    @classmethod
    def _ParseBase(cls, dictionary):
        p = DictionaryParser(cls, dictionary)
        p.Parse('schema_version', default={'version': 1, 'url': ''}, func=SchemaVersion.FromDictionary)
        return p

    def ToDictionary(self):
        w = DictionaryWriter(self)
        w.Write('revision')
        w.Write('release_notes_url')
        w.Write('version')
        w.Write('gcloud_rel_path')
        w.Write('post_processing_command')
        w.Write('schema_version', func=SchemaVersion.ToDictionary)
        w.WriteList('components', func=Component.ToDictionary)
        w.WriteList('notifications', func=NotificationSpec.ToDictionary)
        return w.Dictionary()

    def __init__(self, revision, schema_version, release_notes_url, version, gcloud_rel_path, post_processing_command, components, notifications):
        self.revision = revision
        self.schema_version = schema_version
        self.release_notes_url = release_notes_url
        self.version = version
        self.gcloud_rel_path = gcloud_rel_path
        self.post_processing_command = post_processing_command
        self.components = components
        self.notifications = notifications

    def LastUpdatedString(self):
        try:
            last_updated = config.InstallationConfig.ParseRevision(self.revision)
            return time.strftime('%Y/%m/%d', last_updated)
        except ValueError:
            return 'Unknown'

    def Merge(self, sdk_def):
        current_components = dict(((c.id, c) for c in self.components))
        for c in sdk_def.components:
            if c.id in current_components:
                self.components.remove(current_components[c.id])
                current_components[c.id] = c
            self.components.append(c)