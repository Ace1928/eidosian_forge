from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_catalog import search
def Search(args, version_label):
    """Search Data Catalog for entries, tags, etc that match a query."""
    client = search.SearchClient(version_label)
    return client.Search(args.query, args.include_gcp_public_datasets, args.include_organization_ids, args.restricted_locations, args.include_project_ids, args.order_by, args.page_size, args.limit)