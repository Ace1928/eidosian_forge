import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _do_links(self, text):
    """Turn Markdown link shortcuts into XHTML <a> and <img> tags.

        This is a combination of Markdown.pl's _DoAnchors() and
        _DoImages(). They are done together because that simplified the
        approach. It was necessary to use a different approach than
        Markdown.pl because of the lack of atomic matching support in
        Python's regex engine used in $g_nested_brackets.
        """
    MAX_LINK_TEXT_SENTINEL = 3000
    anchor_allowed_pos = 0
    curr_pos = 0
    while True:
        try:
            start_idx = text.index('[', curr_pos)
        except ValueError:
            break
        text_length = len(text)
        bracket_depth = 0
        for p in range(start_idx + 1, min(start_idx + MAX_LINK_TEXT_SENTINEL, text_length)):
            ch = text[p]
            if ch == ']':
                bracket_depth -= 1
                if bracket_depth < 0:
                    break
            elif ch == '[':
                bracket_depth += 1
        else:
            curr_pos = start_idx + 1
            continue
        link_text = text[start_idx + 1:p]
        if self.safe_mode:
            link_text = self._hash_html_spans(link_text)
            link_text = self._unhash_html_spans(link_text)
        if 'footnotes' in self.extras and link_text.startswith('^'):
            normed_id = re.sub('\\W', '-', link_text[1:])
            if normed_id in self.footnotes:
                self.footnote_ids.append(normed_id)
                result = '<sup class="footnote-ref" id="fnref-%s"><a href="#fn-%s">%s</a></sup>' % (normed_id, normed_id, len(self.footnote_ids))
                text = text[:start_idx] + result + text[p + 1:]
            else:
                curr_pos = p + 1
            continue
        p += 1
        if text[p:p + 1] == '(':
            url, title, url_end_idx = self._extract_url_and_title(text, p)
            if url is not None:
                is_img = start_idx > 0 and text[start_idx - 1] == '!'
                if is_img:
                    start_idx -= 1
                url = url.replace('*', self._escape_table['*']).replace('_', self._escape_table['_'])
                if title:
                    title_str = ' title="%s"' % _xml_escape_attr(title).replace('*', self._escape_table['*']).replace('_', self._escape_table['_'])
                else:
                    title_str = ''
                if is_img:
                    img_class_str = self._html_class_str_from_tag('img')
                    result = '<img src="%s" alt="%s"%s%s%s' % (self._protect_url(url), _xml_escape_attr(link_text), title_str, img_class_str, self.empty_element_suffix)
                    if 'smarty-pants' in self.extras:
                        result = result.replace('"', self._escape_table['"'])
                    curr_pos = start_idx + len(result)
                    anchor_allowed_pos = start_idx + len(result)
                    text = text[:start_idx] + result + text[url_end_idx:]
                elif start_idx >= anchor_allowed_pos:
                    safe_link = self._safe_href.match(url)
                    if self.safe_mode and (not safe_link):
                        result_head = '<a href="#"%s>' % title_str
                    else:
                        result_head = '<a href="%s"%s>' % (self._protect_url(url), title_str)
                    result = '%s%s</a>' % (result_head, link_text)
                    if 'smarty-pants' in self.extras:
                        result = result.replace('"', self._escape_table['"'])
                    curr_pos = start_idx + len(result_head)
                    anchor_allowed_pos = start_idx + len(result)
                    text = text[:start_idx] + result + text[url_end_idx:]
                else:
                    curr_pos = start_idx + 1
                continue
        else:
            match = self._tail_of_reference_link_re.match(text, p)
            if match:
                is_img = start_idx > 0 and text[start_idx - 1] == '!'
                if is_img:
                    start_idx -= 1
                link_id = match.group('id').lower()
                if not link_id:
                    link_id = link_text.lower()
                if link_id in self.urls:
                    url = self.urls[link_id]
                    url = url.replace('*', self._escape_table['*']).replace('_', self._escape_table['_'])
                    title = self.titles.get(link_id)
                    if title:
                        title = _xml_escape_attr(title).replace('*', self._escape_table['*']).replace('_', self._escape_table['_'])
                        title_str = ' title="%s"' % title
                    else:
                        title_str = ''
                    if is_img:
                        img_class_str = self._html_class_str_from_tag('img')
                        result = '<img src="%s" alt="%s"%s%s%s' % (self._protect_url(url), _xml_escape_attr(link_text), title_str, img_class_str, self.empty_element_suffix)
                        if 'smarty-pants' in self.extras:
                            result = result.replace('"', self._escape_table['"'])
                        curr_pos = start_idx + len(result)
                        text = text[:start_idx] + result + text[match.end():]
                    elif start_idx >= anchor_allowed_pos:
                        if self.safe_mode and (not self._safe_href.match(url)):
                            result_head = '<a href="#"%s>' % title_str
                        else:
                            result_head = '<a href="%s"%s>' % (self._protect_url(url), title_str)
                        result = '%s%s</a>' % (result_head, link_text)
                        if 'smarty-pants' in self.extras:
                            result = result.replace('"', self._escape_table['"'])
                        curr_pos = start_idx + len(result_head)
                        anchor_allowed_pos = start_idx + len(result)
                        text = text[:start_idx] + result + text[match.end():]
                    else:
                        curr_pos = start_idx + 1
                else:
                    curr_pos = p
                continue
        curr_pos = start_idx + 1
    return text