from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
from six.moves import map  # pylint: disable=redefined-builtin
from the device. This flag can be specified multiple times to add multiple
def _GetDeviceCredentialFlags(combine_flags=True, only_modifiable=False):
    """"Generates credentials-related flags."""
    flags = []
    if not only_modifiable:
        flags.extend([base.Argument('--path', required=True, type=str, help='The path on disk to the file containing the key.'), base.ChoiceArgument('--type', choices=_VALID_KEY_TYPES, required=True, help_str='The type of the key.')])
    flags.append(base.Argument('--expiration-time', type=arg_parsers.Datetime.Parse, help='The expiration time for the key. See $ gcloud topic datetimes for information on time formats.'))
    if not combine_flags:
        return flags
    sub_argument_help = []
    spec = {}
    for flag in flags:
        name = flag.name.lstrip('-')
        required = flag.kwargs.get('required')
        choices = flag.kwargs.get('choices')
        choices_str = ''
        if choices:
            choices_str = ', '.join(map('`{}`'.format, sorted(choices)))
            choices_str = ' One of [{}].'.format(choices_str)
        help_ = flag.kwargs['help']
        spec[name] = flag.kwargs['type']
        sub_argument_help.append('* *{name}*: {required}.{choices} {help}'.format(name=name, required='Required' if required else 'Optional', choices=choices_str, help=help_))
    key_type_help = []
    for key_type, description in reversed(sorted(_VALID_KEY_TYPES.items())):
        key_type_help.append('* `{}`: {}'.format(key_type, description))
    flag = base.Argument('--public-key', dest='public_keys', metavar='path=PATH,type=TYPE,[expiration-time=EXPIRATION-TIME]', type=arg_parsers.ArgDict(spec=spec), action='append', help='Specify a public key.\n\nSupports four key types:\n\n{key_type_help}\n\nThe key specification is given via the following sub-arguments:\n\n{sub_argument_help}\n\nFor example:\n\n  --public-key \\\n      path=/path/to/id_rsa.pem,type=RSA_PEM,expiration-time=2017-01-01T00:00-05\n\nThis flag may be provide multiple times to provide multiple keys (maximum 3).\n'.format(key_type_help='\n'.join(key_type_help), sub_argument_help='\n'.join(sub_argument_help)))
    return [flag]