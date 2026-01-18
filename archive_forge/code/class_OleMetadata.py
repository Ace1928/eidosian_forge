from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
class OleMetadata:
    """
    Class to parse and store metadata from standard properties of OLE files.

    Available attributes:
    codepage, title, subject, author, keywords, comments, template,
    last_saved_by, revision_number, total_edit_time, last_printed, create_time,
    last_saved_time, num_pages, num_words, num_chars, thumbnail,
    creating_application, security, codepage_doc, category, presentation_target,
    bytes, lines, paragraphs, slides, notes, hidden_slides, mm_clips,
    scale_crop, heading_pairs, titles_of_parts, manager, company, links_dirty,
    chars_with_spaces, unused, shared_doc, link_base, hlinks, hlinks_changed,
    version, dig_sig, content_type, content_status, language, doc_version

    Note: an attribute is set to None when not present in the properties of the
    OLE file.

    References for SummaryInformation stream:

    - https://msdn.microsoft.com/en-us/library/dd942545.aspx
    - https://msdn.microsoft.com/en-us/library/dd925819%28v=office.12%29.aspx
    - https://msdn.microsoft.com/en-us/library/windows/desktop/aa380376%28v=vs.85%29.aspx
    - https://msdn.microsoft.com/en-us/library/aa372045.aspx
    - http://sedna-soft.de/articles/summary-information-stream/
    - https://poi.apache.org/apidocs/org/apache/poi/hpsf/SummaryInformation.html

    References for DocumentSummaryInformation stream:

    - https://msdn.microsoft.com/en-us/library/dd945671%28v=office.12%29.aspx
    - https://msdn.microsoft.com/en-us/library/windows/desktop/aa380374%28v=vs.85%29.aspx
    - https://poi.apache.org/apidocs/org/apache/poi/hpsf/DocumentSummaryInformation.html

    New in version 0.25
    """
    SUMMARY_ATTRIBS = ['codepage', 'title', 'subject', 'author', 'keywords', 'comments', 'template', 'last_saved_by', 'revision_number', 'total_edit_time', 'last_printed', 'create_time', 'last_saved_time', 'num_pages', 'num_words', 'num_chars', 'thumbnail', 'creating_application', 'security']
    DOCSUM_ATTRIBS = ['codepage_doc', 'category', 'presentation_target', 'bytes', 'lines', 'paragraphs', 'slides', 'notes', 'hidden_slides', 'mm_clips', 'scale_crop', 'heading_pairs', 'titles_of_parts', 'manager', 'company', 'links_dirty', 'chars_with_spaces', 'unused', 'shared_doc', 'link_base', 'hlinks', 'hlinks_changed', 'version', 'dig_sig', 'content_type', 'content_status', 'language', 'doc_version']

    def __init__(self):
        """
        Constructor for OleMetadata
        All attributes are set to None by default
        """
        self.codepage = None
        self.title = None
        self.subject = None
        self.author = None
        self.keywords = None
        self.comments = None
        self.template = None
        self.last_saved_by = None
        self.revision_number = None
        self.total_edit_time = None
        self.last_printed = None
        self.create_time = None
        self.last_saved_time = None
        self.num_pages = None
        self.num_words = None
        self.num_chars = None
        self.thumbnail = None
        self.creating_application = None
        self.security = None
        self.codepage_doc = None
        self.category = None
        self.presentation_target = None
        self.bytes = None
        self.lines = None
        self.paragraphs = None
        self.slides = None
        self.notes = None
        self.hidden_slides = None
        self.mm_clips = None
        self.scale_crop = None
        self.heading_pairs = None
        self.titles_of_parts = None
        self.manager = None
        self.company = None
        self.links_dirty = None
        self.chars_with_spaces = None
        self.unused = None
        self.shared_doc = None
        self.link_base = None
        self.hlinks = None
        self.hlinks_changed = None
        self.version = None
        self.dig_sig = None
        self.content_type = None
        self.content_status = None
        self.language = None
        self.doc_version = None

    def parse_properties(self, ole_file):
        """
        Parse standard properties of an OLE file, from the streams
        ``\\x05SummaryInformation`` and ``\\x05DocumentSummaryInformation``,
        if present.
        Properties are converted to strings, integers or python datetime objects.
        If a property is not present, its value is set to None.

        :param ole_file: OleFileIO object from which to parse properties
        """
        for attrib in self.SUMMARY_ATTRIBS + self.DOCSUM_ATTRIBS:
            setattr(self, attrib, None)
        if ole_file.exists('\x05SummaryInformation'):
            props = ole_file.getproperties('\x05SummaryInformation', convert_time=True, no_conversion=[10])
            for i in range(len(self.SUMMARY_ATTRIBS)):
                value = props.get(i + 1, None)
                setattr(self, self.SUMMARY_ATTRIBS[i], value)
        if ole_file.exists('\x05DocumentSummaryInformation'):
            props = ole_file.getproperties('\x05DocumentSummaryInformation', convert_time=True)
            for i in range(len(self.DOCSUM_ATTRIBS)):
                value = props.get(i + 1, None)
                setattr(self, self.DOCSUM_ATTRIBS[i], value)

    def dump(self):
        """
        Dump all metadata, for debugging purposes.
        """
        print('Properties from SummaryInformation stream:')
        for prop in self.SUMMARY_ATTRIBS:
            value = getattr(self, prop)
            print('- {}: {}'.format(prop, repr(value)))
        print('Properties from DocumentSummaryInformation stream:')
        for prop in self.DOCSUM_ATTRIBS:
            value = getattr(self, prop)
            print('- {}: {}'.format(prop, repr(value)))