from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def ParseActiveDirectoryConfig(self, name=None, domain=None, site=None, dns=None, net_bios_prefix=None, organizational_unit=None, aes_encryption=None, username=None, password=None, backup_operators=None, security_operators=None, kdc_hostname=None, kdc_ip=None, nfs_users_with_ldap=None, ldap_signing=None, encrypt_dc_connections=None, description=None, labels=None):
    """Parses the command line arguments for Create Active Directory into a config.

    Args:
      name: the name of the Active Directory
      domain: the domain name of the Active Directory
      site: the site of the Active Directory
      dns: the DNS server IP addresses for the Active Directory domain
      net_bios_prefix: the NetBIOS prefix name of the server
      organizational_unit: The organizational unit within the AD the user
        belongs to
      aes_encryption: Bool, if enabled, AES encryption will be enabled for
        SMB communication
      username: Username of the AD domain admin
      password: Password of the AD domain admin
      backup_operators: The backup operators AD group users list
      security_operators: Security operators AD domain users list
      kdc_hostname: Name of the AD machine
      kdc_ip: KDC Server IP address for the AD machine
      nfs_users_with_ldap: Bool, if enabled, will allow access to local users
        and LDAP users. Disable, if only needed for LDAP users
      ldap_signing: Bool that specifies whether or not LDAP traffic needs to
        be signed
      encrypt_dc_connections: Bool, if enabled, traffic between SMB server
        and DC will be encrypted
      description: the description of the Active Directory
      labels: the labels for the Active Directory

    Returns:
      The configuration that will be used as the request body for creating a
      Cloud NetApp Active Directory.
    """
    active_directory = self.messages.ActiveDirectory()
    active_directory.name = name
    active_directory.domain = domain
    active_directory.site = site
    active_directory.dns = dns
    active_directory.netBiosPrefix = net_bios_prefix
    active_directory.organizationalUnit = organizational_unit
    active_directory.aesEncryption = aes_encryption
    active_directory.username = username
    active_directory.password = password
    active_directory.backupOperators = backup_operators if backup_operators else []
    active_directory.securityOperators = security_operators if security_operators else []
    active_directory.nfsUsersWithLdap = nfs_users_with_ldap
    active_directory.kdcHostname = kdc_hostname
    active_directory.kdcIp = kdc_ip
    active_directory.ldapSigning = ldap_signing
    active_directory.encryptDcConnections = encrypt_dc_connections
    active_directory.description = description
    active_directory.labels = labels
    return active_directory