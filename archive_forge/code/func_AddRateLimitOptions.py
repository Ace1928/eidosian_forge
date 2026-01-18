from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddRateLimitOptions(parser, support_exceed_redirect=True, support_fairshare=False, support_multiple_rate_limit_keys=False):
    """Adds rate limiting related arguments to the argparse."""
    parser.add_argument('--rate-limit-threshold-count', type=int, help='Number of HTTP(S) requests for calculating the threshold for rate limiting requests.')
    parser.add_argument('--rate-limit-threshold-interval-sec', type=int, help='Interval over which the threshold for rate limiting requests is computed.')
    conform_actions = ['allow']
    parser.add_argument('--conform-action', choices=conform_actions, type=lambda x: x.lower(), help='Action to take when requests are under the given threshold. When requests are throttled, this is also the action for all requests which are not dropped.')
    exceed_actions = ['deny-403', 'deny-404', 'deny-429', 'deny-502', 'deny']
    if support_exceed_redirect:
        exceed_actions.append('redirect')
    parser.add_argument('--exceed-action', choices=exceed_actions, type=lambda x: x.lower(), help='      Action to take when requests are above the given threshold. When a request\n      is denied, return the specified HTTP response code. When a request is\n      redirected, use the redirect options based on --exceed-redirect-type and\n      --exceed-redirect-target below.\n      ')
    if support_exceed_redirect:
        exceed_redirect_types = ['google-recaptcha', 'external-302']
        parser.add_argument('--exceed-redirect-type', choices=exceed_redirect_types, type=lambda x: x.lower(), help='        Type for the redirect action that is configured as the exceed action.\n        ')
        parser.add_argument('--exceed-redirect-target', help="        URL target for the redirect action that is configured as the exceed\n        action when the redirect type is ``external-302''.\n        ")
    enforce_on_key = ['ip', 'all', 'http-header', 'xff-ip', 'http-cookie', 'http-path', 'sni', 'region-code', 'tls-ja3-fingerprint', 'user-ip']
    parser.add_argument('--enforce-on-key', choices=enforce_on_key, type=lambda x: x.lower(), help='      Different key types available to enforce the rate limit threshold limit on:' + _RATE_LIMIT_ENFORCE_ON_KEY_TYPES_DESCRIPTION)
    parser.add_argument('--enforce-on-key-name', help='      Determines the key name for the rate limit key. Applicable only for the\n      following rate limit key types:\n      - http-header: The name of the HTTP header whose value is taken as the key\n      value.\n      - http-cookie: The name of the HTTP cookie whose value is taken as the key\n      value.\n      ')
    if support_multiple_rate_limit_keys:
        parser.add_argument('--enforce-on-key-configs', type=arg_parsers.ArgDict(spec={key: str for key in enforce_on_key}, min_length=1, max_length=3, allow_key_only=True), metavar='[[all],[ip],[xff-ip],[http-cookie=HTTP_COOKIE],[http-header=HTTP_HEADER],[http-path],[sni],[region-code],[tls-ja3-fingerprint],[user-ip]]', help='        Specify up to 3 key type/name pairs to rate limit.\n        Valid key types are:\n        ' + _RATE_LIMIT_ENFORCE_ON_KEY_TYPES_DESCRIPTION + '\n      Key names are only applicable to the following key types:\n      - http-header: The name of the HTTP header whose value is taken as the key value.\n      - http-cookie: The name of the HTTP cookie whose value is taken as the key value.\n      ')
    parser.add_argument('--ban-threshold-count', type=int, help="      Number of HTTP(S) requests for calculating the threshold for\n      banning requests. Can only be specified if the action for the\n      rule is ``rate-based-ban''. If specified, the key will be banned\n      for the configured ``BAN_DURATION_SEC'' when the number of requests\n      that exceed the ``RATE_LIMIT_THRESHOLD_COUNT'' also exceed this\n      ``BAN_THRESHOLD_COUNT''.\n      ")
    parser.add_argument('--ban-threshold-interval-sec', type=int, help="      Interval over which the threshold for banning requests is\n      computed. Can only be specified if the action for the rule is\n      ``rate-based-ban''. If specified, the key will be banned for the\n      configured ``BAN_DURATION_SEC'' when the number of requests that\n      exceed the ``RATE_LIMIT_THRESHOLD_COUNT'' also exceed this\n      ``BAN_THRESHOLD_COUNT''.\n      ")
    parser.add_argument('--ban-duration-sec', type=int, help="      Can only be specified if the action for the rule is\n      ``rate-based-ban''. If specified, determines the time (in seconds)\n      the traffic will continue to be banned by the rate limit after\n      the rate falls below the threshold.\n      ")
    if support_fairshare:
        parser.add_argument('--exceed-action-rpc-status-code', type=int, help='Status code, which should be an enum value of [google.rpc.Code]')
        parser.add_argument('--exceed-action-rpc-status-message', help='Developer-facing error message, should be in English.')