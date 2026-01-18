from __future__ import annotations
import copy
from . import common, dates
class OpmlMixin(common.Common):

    def start_opml_opml(self, attrs: dict[str, str]) -> None:
        self.harvest['version'] = 'opml'
        if attrs.get('version') in ('1.0', '1.1'):
            self.harvest['version'] = 'opml1'
        elif attrs.get('version') == '2.0':
            self.harvest['version'] = 'opml2'

    def start_opml_outline(self, attrs: dict[str, str]) -> None:
        if attrs.get('text', '').strip():
            title = attrs['text'].strip()
        else:
            title = attrs.get('title', '').strip()
        url = None
        append_to = None
        if 'xmlurl' in attrs:
            url = attrs.get('xmlurl', '').strip()
            append_to = 'feeds'
            if attrs.get('type', '').strip().lower() == 'source':
                append_to = 'lists'
        elif attrs.get('type', '').lower() in ('link', 'include'):
            append_to = 'lists'
            url = attrs.get('url', '').strip()
        elif title:
            self.hierarchy.append(title)
            return
        if not url and 'htmlurl' in attrs:
            url = attrs['htmlurl'].strip()
            append_to = 'opportunities'
        if not url:
            self.hierarchy.append('')
            return
        if url not in self.found_urls and append_to:
            obj = common.SuperDict({'url': url, 'title': title})
            self.found_urls[url] = (append_to, obj)
            self.harvest[append_to].append(obj)
        else:
            obj = self.found_urls[url][1]
        obj.setdefault('categories', [])
        if 'category' in attrs.keys():
            for i in attrs['category'].split(','):
                tmp = [j.strip() for j in i.split('/') if j.strip()]
                if tmp and tmp not in obj['categories']:
                    obj['categories'].append(tmp)
        if self.hierarchy and self.hierarchy not in obj['categories']:
            obj['categories'].append(copy.copy(self.hierarchy))
        obj['tags'] = [i[0] for i in obj['categories'] if len(i) == 1]
        self.hierarchy.append('')

    def end_opml_outline(self) -> None:
        self.hierarchy.pop()
    start_opml_title = common.Common.expect_text

    def end_opml_title(self) -> None:
        value = self.get_text()
        if value:
            self.harvest['meta']['title'] = value
    start_opml_ownerid = common.Common.expect_text

    def end_opml_ownerid(self) -> None:
        value = self.get_text()
        if value:
            self.harvest['meta'].setdefault('author', common.SuperDict())
            self.harvest['meta']['author']['url'] = value
    start_opml_owneremail = common.Common.expect_text

    def end_opml_owneremail(self) -> None:
        value = self.get_text()
        if value:
            self.harvest['meta'].setdefault('author', common.SuperDict())
            self.harvest['meta']['author']['email'] = value
    start_opml_ownername = common.Common.expect_text

    def end_opml_ownername(self) -> None:
        value = self.get_text()
        if value:
            self.harvest['meta'].setdefault('author', common.SuperDict())
            self.harvest['meta']['author']['name'] = value
    start_opml_datecreated = common.Common.expect_text

    def end_opml_datecreated(self) -> None:
        value = self.get_text()
        if value:
            self.harvest['meta']['created'] = value
            timestamp = dates.parse_rfc822(value)
            if timestamp:
                self.harvest['meta']['created_parsed'] = timestamp
            else:
                self.raise_bozo('dateCreated is not an RFC 822 datetime')
    start_opml_datemodified = common.Common.expect_text

    def end_opml_datemodified(self) -> None:
        value = self.get_text()
        if value:
            self.harvest['meta']['modified'] = value
            timestamp = dates.parse_rfc822(value)
            if timestamp:
                self.harvest['meta']['modified_parsed'] = timestamp
            else:
                self.raise_bozo('dateModified is not an RFC 822 datetime')