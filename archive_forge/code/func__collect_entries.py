import filecmp
import logging
import os
import requests
import wandb
def _collect_entries(art):
    has_next_page = True
    cursor = None
    entries = []
    while has_next_page:
        attrs = art._fetch_file_urls(cursor)
        has_next_page = attrs['pageInfo']['hasNextPage']
        cursor = attrs['pageInfo']['endCursor']
        for edge in attrs['edges']:
            name = edge['node']['name']
            entry = art.get_entry(name)
            entry._download_url = edge['node']['directUrl']
            entries.append(entry)
    return entries