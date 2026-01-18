import abc
import hashlib
import json
import xml.etree.ElementTree as ET  # noqa
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Optional, Tuple, Union
from jupyter_server.base.handlers import APIHandler
from jupyterlab_server.translation_utils import translator
from packaging.version import parse
from tornado import httpclient, web
from jupyterlab._version import __version__
class NewsHandler(APIHandler):
    """News API handler.

    Args:
        news_url: The Atom feed to fetch for news
    """

    def initialize(self, news_url: Optional[str]=None) -> None:
        super().initialize()
        self.news_url = news_url

    @web.authenticated
    async def get(self):
        """Get the news.

        Response:
            {
                "news": List[Notification]
            }
        """
        news = []
        http_client = httpclient.AsyncHTTPClient()
        if self.news_url is not None:
            trans = translator.load('jupyterlab')
            xml_namespaces = {'atom': 'http://www.w3.org/2005/Atom'}
            for key, spec in xml_namespaces.items():
                ET.register_namespace(key, spec)
            try:
                response = await http_client.fetch(self.news_url, headers={'Content-Type': 'application/atom+xml'})
                tree = ET.fromstring(response.body)

                def build_entry(node):

                    def get_xml_text(attr: str, default: Optional[str]=None) -> str:
                        node_item = node.find(f'atom:{attr}', xml_namespaces)
                        if node_item is not None:
                            return node_item.text
                        elif default is not None:
                            return default
                        else:
                            error_m = f'atom feed entry does not contain a required attribute: {attr}'
                            raise KeyError(error_m)
                    entry_title = get_xml_text('title')
                    entry_id = get_xml_text('id')
                    entry_updated = get_xml_text('updated')
                    entry_published = get_xml_text('published', entry_updated)
                    entry_summary = get_xml_text('summary', default='')
                    links = node.findall('atom:link', xml_namespaces)
                    if len(links) > 1:
                        alternate = list(filter(lambda elem: elem.get('rel') == 'alternate', links))
                        link_node = alternate[0] if alternate else links[0]
                    else:
                        link_node = links[0] if len(links) == 1 else None
                    entry_link = link_node.get('href') if link_node is not None else None
                    message = '\n'.join([entry_title, entry_summary]) if entry_summary else entry_title
                    modified_at = format_datetime(entry_updated)
                    created_at = format_datetime(entry_published)
                    notification = Notification(message=message, createdAt=created_at, modifiedAt=modified_at, type='info', link=None if entry_link is None else (trans.__('Open full post'), entry_link), options={'data': {'id': entry_id, 'tags': ['news']}})
                    return notification
                entries = map(build_entry, tree.findall('atom:entry', xml_namespaces))
                news.extend(entries)
            except Exception as e:
                self.log.debug(f'Failed to get announcements from Atom feed: {self.news_url}', exc_info=e)
        self.set_status(200)
        self.finish(json.dumps({'news': list(map(asdict, news))}))