import html
from os import path
from typing import Any, Dict, List, NamedTuple, Set, Tuple
from sphinx import package_dir
from sphinx.application import Sphinx
from sphinx.builders import _epub_base
from sphinx.config import ENUM, Config
from sphinx.locale import __
from sphinx.util import logging, xmlname_checker
from sphinx.util.fileutil import copy_asset_file
from sphinx.util.i18n import format_date
from sphinx.util.osutil import make_filename
class Epub3Builder(_epub_base.EpubBuilder):
    """
    Builder that outputs epub3 files.

    It creates the metainfo files content.opf, nav.xhtml, toc.ncx, mimetype,
    and META-INF/container.xml. Afterwards, all necessary files are zipped to
    an epub file.
    """
    name = 'epub'
    epilog = __('The ePub file is in %(outdir)s.')
    supported_remote_images = False
    template_dir = path.join(package_dir, 'templates', 'epub3')
    doctype = DOCTYPE
    html_tag = HTML_TAG
    use_meta_charset = True

    def handle_finish(self) -> None:
        """Create the metainfo files and finally the epub."""
        self.get_toc()
        self.build_mimetype()
        self.build_container()
        self.build_content()
        self.build_navigation_doc()
        self.build_toc()
        self.build_epub()

    def content_metadata(self) -> Dict[str, Any]:
        """Create a dictionary with all metadata for the content.opf
        file properly escaped.
        """
        writing_mode = self.config.epub_writing_mode
        metadata = super().content_metadata()
        metadata['description'] = html.escape(self.config.epub_description)
        metadata['contributor'] = html.escape(self.config.epub_contributor)
        metadata['page_progression_direction'] = PAGE_PROGRESSION_DIRECTIONS.get(writing_mode)
        metadata['ibook_scroll_axis'] = IBOOK_SCROLL_AXIS.get(writing_mode)
        metadata['date'] = html.escape(format_date('%Y-%m-%dT%H:%M:%SZ', language='en'))
        metadata['version'] = html.escape(self.config.version)
        metadata['epub_version'] = self.config.epub_version
        return metadata

    def prepare_writing(self, docnames: Set[str]) -> None:
        super().prepare_writing(docnames)
        writing_mode = self.config.epub_writing_mode
        self.globalcontext['theme_writing_mode'] = THEME_WRITING_MODES.get(writing_mode)
        self.globalcontext['html_tag'] = self.html_tag
        self.globalcontext['use_meta_charset'] = self.use_meta_charset
        self.globalcontext['skip_ua_compatible'] = True

    def build_navlist(self, navnodes: List[Dict[str, Any]]) -> List[NavPoint]:
        """Create the toc navigation structure.

        This method is almost same as build_navpoints method in epub.py.
        This is because the logical navigation structure of epub3 is not
        different from one of epub2.

        The difference from build_navpoints method is templates which are used
        when generating navigation documents.
        """
        navstack: List[NavPoint] = []
        navstack.append(NavPoint('', '', []))
        level = 0
        for node in navnodes:
            if not node['text']:
                continue
            file = node['refuri'].split('#')[0]
            if file in self.ignored_files:
                continue
            if node['level'] > self.config.epub_tocdepth:
                continue
            navpoint = NavPoint(node['text'], node['refuri'], [])
            if node['level'] == level:
                navstack.pop()
                navstack[-1].children.append(navpoint)
                navstack.append(navpoint)
            elif node['level'] == level + 1:
                level += 1
                navstack[-1].children.append(navpoint)
                navstack.append(navpoint)
            elif node['level'] < level:
                while node['level'] < len(navstack):
                    navstack.pop()
                level = node['level']
                navstack[-1].children.append(navpoint)
                navstack.append(navpoint)
            else:
                raise RuntimeError('Should never reach here. It might be a bug.')
        return navstack[0].children

    def navigation_doc_metadata(self, navlist: List[NavPoint]) -> Dict[str, Any]:
        """Create a dictionary with all metadata for the nav.xhtml file
        properly escaped.
        """
        metadata = {}
        metadata['lang'] = html.escape(self.config.epub_language)
        metadata['toc_locale'] = html.escape(self.guide_titles['toc'])
        metadata['navlist'] = navlist
        return metadata

    def build_navigation_doc(self) -> None:
        """Write the metainfo file nav.xhtml."""
        logger.info(__('writing nav.xhtml file...'))
        if self.config.epub_tocscope == 'default':
            doctree = self.env.get_and_resolve_doctree(self.config.root_doc, self, prune_toctrees=False, includehidden=False)
            refnodes = self.get_refnodes(doctree, [])
            self.toc_add_files(refnodes)
        else:
            refnodes = self.refnodes
        navlist = self.build_navlist(refnodes)
        copy_asset_file(path.join(self.template_dir, 'nav.xhtml_t'), self.outdir, self.navigation_doc_metadata(navlist))
        if 'nav.xhtml' not in self.files:
            self.files.append('nav.xhtml')