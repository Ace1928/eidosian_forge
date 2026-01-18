from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def GetCrawlScopeArg(for_update):
    choices = {'bucket': 'Directs the crawler to crawl specific buckets within the project that owns the crawler.', 'project': 'Directs the crawler to crawl all the buckets of the project that owns the crawler.', 'organization': 'Directs the crawler to crawl all the buckets of the projects in the organization that owns the crawler.'}
    return base.ChoiceArgument('--crawl-scope', choices=choices, required=not for_update, help_str='Scope of the crawler.')