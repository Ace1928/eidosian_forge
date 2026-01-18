from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import contextlib
import enum
from functools import wraps  # pylint:disable=g-importing-member
import itertools
import re
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import display
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_printer
import six
class _Common(six.with_metaclass(abc.ABCMeta, object)):
    """Base class for Command and Group."""
    category = None
    _cli_generator = None
    _is_hidden = False
    _is_unicode_supported = False
    _release_track = None
    _universe_compatible = None
    _valid_release_tracks = None
    _notices = None
    _is_deprecated = False

    def __init__(self, is_group=False):
        self.exit_code = 0
        self.is_group = is_group

    @staticmethod
    def Args(parser):
        """Set up arguments for this command.

    Args:
      parser: An argparse.ArgumentParser.
    """
        pass

    @staticmethod
    def _Flags(parser):
        """Adds subclass flags.

    Args:
      parser: An argparse.ArgumentParser object.
    """
        pass

    @classmethod
    def IsHidden(cls):
        return cls._is_hidden

    @classmethod
    def IsUniverseCompatible(cls):
        return cls._universe_compatible

    @classmethod
    def IsUnicodeSupported(cls):
        if six.PY2:
            return cls._is_unicode_supported
        return True

    @classmethod
    def ReleaseTrack(cls):
        return cls._release_track

    @classmethod
    def ValidReleaseTracks(cls):
        return cls._valid_release_tracks

    @classmethod
    def GetTrackedAttribute(cls, obj, attribute):
        """Gets the attribute value from obj for tracks.

    The values are checked in ReleaseTrack order.

    Args:
      obj: The object to extract attribute from.
      attribute: The attribute name in object.

    Returns:
      The attribute value from obj for tracks.
    """
        for track in ReleaseTrack:
            if track not in cls._valid_release_tracks:
                continue
            names = []
            names.append(attribute + '_' + track.id)
            if track.prefix:
                names.append(attribute + '_' + track.prefix)
            for name in names:
                if hasattr(obj, name):
                    return getattr(obj, name)
        return getattr(obj, attribute, None)

    @classmethod
    def Notices(cls):
        return cls._notices

    @classmethod
    def AddNotice(cls, tag, msg, preserve_existing=False):
        if not cls._notices:
            cls._notices = {}
        if tag in cls._notices and preserve_existing:
            return
        cls._notices[tag] = msg

    @classmethod
    def Deprecated(cls):
        return cls._is_deprecated

    @classmethod
    def SetDeprecated(cls, is_deprecated):
        cls._is_deprecated = is_deprecated

    @classmethod
    def GetCLIGenerator(cls):
        """Get a generator function that can be used to execute a gcloud command.

    Returns:
      A bound generator function to execute a gcloud command.
    """
        if cls._cli_generator:
            return cls._cli_generator.Generate
        return None

    @classmethod
    def EnableSelfSignedJwtForTracks(cls, tracks):
        """Enable self signed jwt feature for the given tracks.

    The feature can be disabled manually by running
    `gcloud config set auth/service_account_use_self_signed_jwt false`.

    Args:
      tracks: [base.ReleaseTrack], A list of release tracks where self signed
        jwt feature is enabled.
    """
        if properties.VALUES.auth.service_account_use_self_signed_jwt.IsExplicitlySet():
            return
        if cls.ReleaseTrack() and cls.ReleaseTrack() in tracks:
            properties.VALUES.auth.service_account_use_self_signed_jwt.Set(True)