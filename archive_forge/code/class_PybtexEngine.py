from __future__ import absolute_import
from __future__ import unicode_literals
from os import path
class PybtexEngine(Engine):
    """
    The Python fomatting engine.

    See :py:class:`pybtex.Engine` for inherited methods.
    """

    def format_from_files(self, bib_files_or_filenames, style, citations=['*'], bib_format=None, bib_encoding=None, output_backend=None, output_encoding=None, min_crossrefs=2, output_filename=None, add_output_suffix=False, **kwargs):
        """
        Read the bigliography data from the given files and produce a formated
        bibliography.

        :param bib_files_or_filenames: A list of file names or file objects.
        :param style: The name of the formatting style.
        :param citations: A list of citation keys.
        :param bib_format: The name of the bibliography format. The default
            format is ``bibtex``.
        :param bib_encoding: Encoding of bibliography files.
        :param output_backend: Which output backend to use. The default is ``latex``.
        :param output_encoding: Encoding that will be used by the output backend.
        :param bst_encoding: Encoding of the ``.bst`` file.
        :param min_crossrefs: Include cross-referenced entries after this many
            crossrefs. See BibTeX manual for details.
        :param output_filename: If ``None``, the result will be returned as a
            string. Else, the result will be written to the specified file.
        :param add_output_suffix: Append default suffix to the output file
            name (``.bbl`` for LaTeX, ``.html`` for HTML, etc.).
        """
        from pybtex.plugin import find_plugin
        bib_parser = find_plugin('pybtex.database.input', bib_format)
        bib_data = bib_parser(encoding=bib_encoding, wanted_entries=citations, min_crossrefs=min_crossrefs).parse_files(bib_files_or_filenames)
        style_cls = find_plugin('pybtex.style.formatting', style)
        style = style_cls(label_style=kwargs.get('label_style'), name_style=kwargs.get('name_style'), sorting_style=kwargs.get('sorting_style'), abbreviate_names=kwargs.get('abbreviate_names'), min_crossrefs=min_crossrefs)
        formatted_bibliography = style.format_bibliography(bib_data, citations)
        output_backend = find_plugin('pybtex.backends', output_backend)
        if add_output_suffix:
            output_filename = output_filename + output_backend.default_suffix
        if not output_filename:
            import io
            output_filename = io.StringIO()
        return output_backend(output_encoding).write_to_file(formatted_bibliography, output_filename)