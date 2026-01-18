from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import enum
import errno
import getpass
import os
import re
import string
import subprocess
import tempfile
import textwrap
from googlecloudsdk.api_lib.oslogin import client as oslogin_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.oslogin import oslogin_utils
from googlecloudsdk.command_lib.util import gaia
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import retry
import six
from six.moves.urllib.parse import quote
def GetOsloginState(instance, project, requested_user, public_key, expiration_time, release_track, username_requested=False, instance_enable_oslogin=None, instance_enable_2fa=None, instance_enable_security_keys=None, instance_require_certificates=None, messages=None):
    """Check instance/project metadata for oslogin and return updated username.

  Check to see if OS Login is enabled in metadata and if it is, return
  the OS Login user and a boolean value indicating if OS Login is being used.

  Args:
    instance: instance, The object representing the instance we are connecting
      to. If None, instance metadata will be ignored.
    project: project, The object representing the current project.
    requested_user: str, The default or requested username to connect as.
    public_key: str, The public key of the user connecting.
    expiration_time: int, Microseconds after epoch when the ssh key should
      expire. If None, an existing key will not be modified and a new key will
      not be set to expire.  If not None, an existing key may be modified with
      the new expiry.
    release_track: release_track, The object representing the release track.
    username_requested: bool, True if the user has passed a specific username in
      the args.
    instance_enable_oslogin: True if the instance's metadata indicates that OS
      Login is enabled, and False if not enabled. Used when the instance cannot
      be passed through the instance argument. None if not specified.
    instance_enable_2fa: True if the instance's metadata indicates that OS Login
      2FA is enabled, and False if not enabled. Used when the instance cannot be
      passed through the instance argument. None if not specified.
    instance_enable_security_keys: True if the instance's metadata indicates
      that OS Login security keys are enabled, and False if not enabled. Used
      when the instance cannot be passed through the instance argument. None if
      not specified.
    instance_require_certificates: True if the instance's metadata indicates
      that OS Login SSH certificates are to be used exclusively, False
      otherwise. An override to be used when the instance cannot be passed
      through the "instance" argument. None if not specified.
    messages: API messages class, The compute API messages.

  Returns:
    object, An object containing the OS Login state, with values indicating
      whether OS Login is enabled, Security Keys are enabled, the username to
      connect as and a list of security keys.
  """
    oslogin_state = OsloginState(user=requested_user)
    oslogin_enabled = FeatureEnabledInMetadata(instance, project, OSLOGIN_ENABLE_METADATA_KEY, instance_override=instance_enable_oslogin)
    if not oslogin_enabled:
        return oslogin_state
    if _IsInstanceWindows(instance, messages=messages):
        log.status.Print('OS Login is not available on Windows VMs.\nUsing ssh metadata.')
        return oslogin_state
    oslogin_state.oslogin_enabled = oslogin_enabled
    oslogin_state.oslogin_2fa_enabled = FeatureEnabledInMetadata(instance, project, OSLOGIN_ENABLE_2FA_METADATA_KEY, instance_override=instance_enable_2fa)
    oslogin_state.security_keys_enabled = FeatureEnabledInMetadata(instance, project, OSLOGIN_ENABLE_SK_METADATA_KEY, instance_override=instance_enable_security_keys)
    oslogin_state.third_party_user = IsThirdPartyUser()
    oslogin_state.require_certificates = FeatureEnabledInMetadata(instance, project, OSLOGIN_ENABLE_CERTIFICATES_METADATA_KEY, instance_override=instance_require_certificates)
    env = Environment.Current()
    if env.suite == Suite.PUTTY:
        oslogin_state.environment = 'putty'
    else:
        oslogin_state.environment = 'ssh'
    if oslogin_state.security_keys_enabled:
        oslogin_state.ssh_security_key_support = CheckSshSecurityKeySupport()
    oslogin = oslogin_client.OsloginClient(release_track)
    user_email = properties.VALUES.auth.impersonate_service_account.Get() or properties.VALUES.core.account.Get()
    if release_track in [base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA] and oslogin_state.third_party_user or oslogin_state.require_certificates:
        if oslogin_state.third_party_user:
            user_email = quote(user_email, safe=':')
        zone = instance.zone.split('/').pop()
        region = zone[:zone.rindex('-')]
        ValidateCertificate(oslogin_state, region)
        if not oslogin_state.signed_ssh_key:
            sign_response = oslogin.SignSshPublicKey(user_email, public_key, project.name, region)
            WriteCertificate(region, sign_response.signedSshPublicKey)
        login_profile = oslogin.GetLoginProfile(user_email, project.name, include_security_keys=oslogin_state.security_keys_enabled)
    else:
        login_profile = oslogin.GetLoginProfile(user_email, project.name, include_security_keys=oslogin_state.security_keys_enabled)
        if oslogin_state.security_keys_enabled:
            oslogin_state.security_keys = oslogin_utils.GetSecurityKeysFromProfile(user_email, oslogin, profile=login_profile)
            if not login_profile.posixAccounts:
                import_response = oslogin.ImportSshPublicKey(user_email, '')
                login_profile = import_response.loginProfile
        else:
            keys = oslogin_utils.GetKeyDictionaryFromProfile(user_email, oslogin, profile=login_profile)
            fingerprint = oslogin_utils.FindKeyInKeyList(public_key, keys)
            if not fingerprint or not login_profile.posixAccounts:
                import_response = oslogin.ImportSshPublicKey(user_email, public_key, expiration_time)
                login_profile = import_response.loginProfile
            elif expiration_time:
                oslogin.UpdateSshPublicKey(user_email, fingerprint, keys[fingerprint], 'expirationTimeUsec', expiration_time=expiration_time)
    oslogin_user = None
    for pa in login_profile.posixAccounts:
        oslogin_user = oslogin_user or pa.username
        if pa.username == requested_user:
            return oslogin_state
        elif pa.primary:
            oslogin_user = pa.username
    oslogin_state.user = oslogin_user
    if username_requested:
        log.status.Print('Using OS Login user [{0}] instead of requested user [{1}]'.format(oslogin_user, requested_user))
    else:
        log.info('Using OS Login user [{0}] instead of default user [{1}]'.format(oslogin_user, requested_user))
    return oslogin_state