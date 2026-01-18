from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import exceptions as calliope_exc
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def SetUptimeCheckMatcherFields(args, messages, uptime_check):
    """Set Matcher fields based on args."""
    if args.IsSpecified('matcher_content'):
        if uptime_check.syntheticMonitor is not None:
            raise calliope_exc.InvalidArgumentException('--matcher_content', 'Should not be set for Synthetic Monitor.')
        content_matcher = messages.ContentMatcher()
        content_matcher.content = args.matcher_content
        matcher_mapping = {'contains-string': messages.ContentMatcher.MatcherValueValuesEnum.CONTAINS_STRING, 'not-contains-string': messages.ContentMatcher.MatcherValueValuesEnum.NOT_CONTAINS_STRING, 'matches-regex': messages.ContentMatcher.MatcherValueValuesEnum.MATCHES_REGEX, 'not-matches-regex': messages.ContentMatcher.MatcherValueValuesEnum.NOT_MATCHES_REGEX, 'matches-json-path': messages.ContentMatcher.MatcherValueValuesEnum.MATCHES_JSON_PATH, 'not-matches-json-path': messages.ContentMatcher.MatcherValueValuesEnum.NOT_MATCHES_JSON_PATH, None: messages.ContentMatcher.MatcherValueValuesEnum.CONTAINS_STRING}
        content_matcher.matcher = matcher_mapping.get(args.matcher_type)
        if args.IsSpecified('json_path'):
            if content_matcher.matcher not in (messages.ContentMatcher.MatcherValueValuesEnum.MATCHES_JSON_PATH, messages.ContentMatcher.MatcherValueValuesEnum.NOT_MATCHES_JSON_PATH):
                raise calliope_exc.InvalidArgumentException('--json-path', 'Should only be used with JSON_PATH matcher types.')
            content_matcher.jsonPathMatcher = messages.JsonPathMatcher()
            content_matcher.jsonPathMatcher.jsonPath = args.json_path
            jsonpath_matcher_mapping = {'exact-match': messages.JsonPathMatcher.JsonMatcherValueValuesEnum.EXACT_MATCH, 'regex-match': messages.JsonPathMatcher.JsonMatcherValueValuesEnum.REGEX_MATCH, None: messages.JsonPathMatcher.JsonMatcherValueValuesEnum.EXACT_MATCH}
            content_matcher.jsonPathMatcher.jsonMatcher = jsonpath_matcher_mapping.get(args.json_path_matcher_type)
        uptime_check.contentMatchers = []
        uptime_check.contentMatchers.append(content_matcher)