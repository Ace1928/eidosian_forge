from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.domains import contacts_util
from googlecloudsdk.command_lib.domains import flags
from googlecloudsdk.command_lib.domains import resource_args
from googlecloudsdk.command_lib.domains import util
from googlecloudsdk.core import log
Configure contact settings of a Cloud Domains registration.

  Configure registration's contact settings: email, phone number, postal
  address and also contact privacy.

  In some cases such changes have to be confirmed through an email sent to
  the registrant before they take effect. In order to resend the email, execute
  this command again.

  NOTE: Please consider carefully any changes to contact privacy settings when
  changing from "redacted-contact-data" to "public-contact-data."
  There may be a delay in reflecting updates you make to registrant
  contact information such that any changes you make to contact privacy
  (including from "redacted-contact-data" to "public-contact-data")
  will be applied without delay but changes to registrant contact
  information may take a limited time to be publicized. This means that
  changes to contact privacy from "redacted-contact-data" to
  "public-contact-data" may make the previous registrant contact
  data public until the modified registrant contact details are published.

  ## EXAMPLES

  To start an interactive flow to configure contact settings for
  ``example.com'', run:

    $ {command} example.com

  To enable contact privacy for ``example.com'', run:

    $ {command} example.com --contact-privacy=private-contact-data

  To change contact data for ``example.com'' according to information from a
  YAML file ``contacts.yaml'', run:

    $ {command} example.com --contact-data-from-file=contacts.yaml
  