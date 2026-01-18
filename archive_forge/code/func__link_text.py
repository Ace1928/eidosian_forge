import copy
import re
from urllib.parse import urlsplit, unquote_plus
from lxml import etree
from lxml.html import defs
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import xhtml_to_html, _transform_result
def _link_text(text, link_regexes, avoid_hosts, factory):
    leading_text = ''
    links = []
    last_pos = 0
    while 1:
        best_match, best_pos = (None, None)
        for regex in link_regexes:
            regex_pos = last_pos
            while 1:
                match = regex.search(text, pos=regex_pos)
                if match is None:
                    break
                host = match.group('host')
                for host_regex in avoid_hosts:
                    if host_regex.search(host):
                        regex_pos = match.end()
                        break
                else:
                    break
            if match is None:
                continue
            if best_pos is None or match.start() < best_pos:
                best_match = match
                best_pos = match.start()
        if best_match is None:
            if links:
                assert not links[-1].tail
                links[-1].tail = text
            else:
                assert not leading_text
                leading_text = text
            break
        link = best_match.group(0)
        end = best_match.end()
        if link.endswith('.') or link.endswith(','):
            end -= 1
            link = link[:-1]
        prev_text = text[:best_match.start()]
        if links:
            assert not links[-1].tail
            links[-1].tail = prev_text
        else:
            assert not leading_text
            leading_text = prev_text
        anchor = factory('a')
        anchor.set('href', link)
        body = best_match.group('body')
        if not body:
            body = link
        if body.endswith('.') or body.endswith(','):
            body = body[:-1]
        anchor.text = body
        links.append(anchor)
        text = text[end:]
    return (leading_text, links)