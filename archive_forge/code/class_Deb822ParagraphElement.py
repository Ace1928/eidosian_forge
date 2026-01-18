import collections.abc
import contextlib
import sys
import textwrap
import weakref
from abc import ABC
from types import TracebackType
from weakref import ReferenceType
from debian._deb822_repro._util import (combine_into_replacement, BufferingIterator,
from debian._deb822_repro.formatter import (
from debian._deb822_repro.tokens import (
from debian._deb822_repro.types import AmbiguousDeb822FieldKeyError, SyntaxOrParseError
from debian._util import (
class Deb822ParagraphElement(Deb822Element, Deb822ParagraphToStrWrapperMixin, ABC):

    @classmethod
    def new_empty_paragraph(cls):
        return Deb822NoDuplicateFieldsParagraphElement([], OrderedSet())

    @classmethod
    def from_dict(cls, mapping):
        paragraph = cls.new_empty_paragraph()
        for k, v in mapping.items():
            paragraph[k] = v
        return paragraph

    @classmethod
    def from_kvpairs(cls, kvpair_elements):
        if not kvpair_elements:
            raise ValueError('A paragraph must consist of at least one field/value pair')
        kvpair_order = OrderedSet((kv.field_name for kv in kvpair_elements))
        if len(kvpair_order) == len(kvpair_elements):
            return Deb822NoDuplicateFieldsParagraphElement(kvpair_elements, kvpair_order)
        return Deb822DuplicateFieldsParagraphElement(kvpair_elements)

    @property
    def has_duplicate_fields(self):
        """Tell whether this paragraph has duplicate fields"""
        return False

    def as_interpreted_dict_view(self, interpretation, *, auto_resolve_ambiguous_fields=True):
        """Provide a Dict-like view of the paragraph

        This method returns a dict-like object representing this paragraph and
        is useful for accessing fields in a given interpretation. It is possible
        to use multiple versions of this dict-like view with different interpretations
        on the same paragraph at the same time (for different fields).

            >>> example_deb822_paragraph = '''
            ... Package: foo
            ... # Field comment (because it becomes just before a field)
            ... Architecture: amd64
            ... # Inline comment (associated with the next line)
            ...               i386
            ... # We also support arm
            ...               arm64
            ...               armel
            ... '''
            >>> dfile = parse_deb822_file(example_deb822_paragraph.splitlines())
            >>> paragraph = next(iter(dfile))
            >>> list_view = paragraph.as_interpreted_dict_view(LIST_SPACE_SEPARATED_INTERPRETATION)
            >>> # With the defaults, you only deal with the semantic values
            >>> # - no leading or trailing whitespace on the first part of the value
            >>> list(list_view["Package"])
            ['foo']
            >>> with list_view["Architecture"] as arch_list:
            ...     orig_arch_list = list(arch_list)
            ...     arch_list.replace('i386', 'kfreebsd-amd64')
            >>> orig_arch_list
            ['amd64', 'i386', 'arm64', 'armel']
            >>> list(list_view["Architecture"])
            ['amd64', 'kfreebsd-amd64', 'arm64', 'armel']
            >>> print(paragraph.dump(), end='')
            Package: foo
            # Field comment (because it becomes just before a field)
            Architecture: amd64
            # Inline comment (associated with the next line)
                          kfreebsd-amd64
            # We also support arm
                          arm64
                          armel
            >>> # Format preserved and architecture replaced
            >>> with list_view["Architecture"] as arch_list:
            ...     # Prettify the result as sorting will cause awkward whitespace
            ...     arch_list.reformat_when_finished()
            ...     arch_list.sort()
            >>> print(paragraph.dump(), end='')
            Package: foo
            # Field comment (because it becomes just before a field)
            Architecture: amd64
            # We also support arm
                          arm64
                          armel
            # Inline comment (associated with the next line)
                          kfreebsd-amd64
            >>> list(list_view["Architecture"])
            ['amd64', 'arm64', 'armel', 'kfreebsd-amd64']
            >>> # Format preserved and architecture values sorted

        :param interpretation: Decides how the field values are interpreted.  As an example,
          use LIST_SPACE_SEPARATED_INTERPRETATION for fields such as Architecture in the
          debian/control file.
        :param auto_resolve_ambiguous_fields: This parameter is only relevant for paragraphs
          that contain the same field multiple times (these are generally invalid).  If the
          caller requests an ambiguous field from an invalid paragraph via a plain field name,
          the return dict-like object will refuse to resolve the field (not knowing which
          version to pick).  This parameter (if set to True) instead changes the error into
          assuming the caller wants the *first* variant.
        """
        return Deb822InterpretingParagraphWrapper(self, interpretation, auto_resolve_ambiguous_fields=auto_resolve_ambiguous_fields)

    def configured_view(self, *, discard_comments_on_read=True, auto_map_initial_line_whitespace=True, auto_resolve_ambiguous_fields=True, preserve_field_comments_on_field_updates=True, auto_map_final_newline_in_multiline_values=True):
        """Provide a Dict[str, str]-like view of this paragraph with non-standard parameters

        This method returns a dict-like object representing this paragraph that is
        optionally configured differently from the default view.

            >>> example_deb822_paragraph = '''
            ... Package: foo
            ... # Field comment (because it becomes just before a field)
            ... Depends: libfoo,
            ... # Inline comment (associated with the next line)
            ...          libbar,
            ... '''
            >>> dfile = parse_deb822_file(example_deb822_paragraph.splitlines())
            >>> paragraph = next(iter(dfile))
            >>> # With the defaults, you only deal with the semantic values
            >>> # - no leading or trailing whitespace on the first part of the value
            >>> paragraph["Package"]
            'foo'
            >>> # - no inline comments in multiline values (but whitespace will be present
            >>> #   subsequent lines.)
            >>> print(paragraph["Depends"])
            libfoo,
                     libbar,
            >>> paragraph['Foo'] = 'bar'
            >>> paragraph.get('Foo')
            'bar'
            >>> paragraph.get('Unknown-Field') is None
            True
            >>> # But you get asymmetric behaviour with set vs. get
            >>> paragraph['Foo'] = ' bar\\n'
            >>> paragraph['Foo']
            'bar'
            >>> paragraph['Bar'] = '     bar\\n#Comment\\n another value\\n'
            >>> # Note that the whitespace on the first line has been normalized.
            >>> print("Bar: " + paragraph['Bar'])
            Bar: bar
             another value
            >>> # The comment is present (in case you where wondering)
            >>> print(paragraph.get_kvpair_element('Bar').convert_to_text(), end='')
            Bar:     bar
            #Comment
             another value
            >>> # On the other hand, you can choose to see the values as they are
            >>> # - We will just reset the paragraph as a "nothing up my sleeve"
            >>> dfile = parse_deb822_file(example_deb822_paragraph.splitlines())
            >>> paragraph = next(iter(dfile))
            >>> nonstd_dictview = paragraph.configured_view(
            ...     discard_comments_on_read=False,
            ...     auto_map_initial_line_whitespace=False,
            ...     # For paragraphs with duplicate fields, you can choose to get an error
            ...     # rather than the dict picking the first value available.
            ...     auto_resolve_ambiguous_fields=False,
            ...     auto_map_final_newline_in_multiline_values=False,
            ... )
            >>> # Because we have reset the state, Foo and Bar are no longer there.
            >>> 'Bar' not in paragraph and 'Foo' not in paragraph
            True
            >>> # We can now see the comments (discard_comments_on_read=False)
            >>> # (The leading whitespace in front of "libfoo" is due to
            >>> #  auto_map_initial_line_whitespace=False)
            >>> print(nonstd_dictview["Depends"], end='')
             libfoo,
            # Inline comment (associated with the next line)
                     libbar,
            >>> # And all the optional whitespace on the first value line
            >>> # (auto_map_initial_line_whitespace=False)
            >>> nonstd_dictview["Package"] == ' foo\\n'
            True
            >>> # ... which will give you symmetric behaviour with set vs. get
            >>> nonstd_dictview['Foo'] = '  bar \\n'
            >>> nonstd_dictview['Foo']
            '  bar \\n'
            >>> nonstd_dictview['Bar'] = '  bar \\n#Comment\\n another value\\n'
            >>> nonstd_dictview['Bar']
            '  bar \\n#Comment\\n another value\\n'
            >>> # But then you get no help either.
            >>> try:
            ...     nonstd_dictview["Baz"] = "foo"
            ... except ValueError:
            ...     print("Rejected")
            Rejected
            >>> # With auto_map_initial_line_whitespace=False, you have to include minimum a newline
            >>> nonstd_dictview["Baz"] = "foo\\n"
            >>> # The absence of leading whitespace gives you the terse variant at the expensive
            >>> # readability
            >>> paragraph.get_kvpair_element('Baz').convert_to_text()
            'Baz:foo\\n'
            >>> # But because they are views, changes performed via one view is visible in the other
            >>> paragraph['Foo']
            'bar'
            >>> # The views show the values according to their own rules. Therefore, there is an
            >>> # asymmetric between paragraph['Foo'] and nonstd_dictview['Foo']
            >>> # Nevertheless, you can read or write the fields via either - enabling you to use
            >>> # the view that best suit your use-case for the given field.
            >>> 'Baz' in paragraph and nonstd_dictview.get('Baz') is not None
            True
            >>> # Deletion via the view also works
            >>> del nonstd_dictview['Baz']
            >>> 'Baz' not in paragraph and nonstd_dictview.get('Baz') is None
            True


        :param discard_comments_on_read: When getting a field value from the dict,
          this parameter decides how in-line comments are handled.  When setting
          the value, inline comments are still allowed and will be retained.
          However, keep in mind that this option makes getter and setter assymetric
          as a "get" following a "set" with inline comments will omit the comments
          even if they are there (see the code example).
        :param auto_map_initial_line_whitespace: Special-case the first value line
          by trimming unnecessary whitespace leaving only the value. For single-line
          values, all space including newline is pruned. For multi-line values, the
          newline is preserved / needed to distinguish the first line from the
          following lines.  When setting a value, this option normalizes the
          whitespace of the initial line of the value field.
          When this option is set to True makes the dictionary behave more like the
          original Deb822 module.
        :param preserve_field_comments_on_field_updates: Whether to preserve the field
          comments when mutating the field.
        :param auto_resolve_ambiguous_fields: This parameter is only relevant for paragraphs
          that contain the same field multiple times (these are generally invalid).  If the
          caller requests an ambiguous field from an invalid paragraph via a plain field name,
          the return dict-like object will refuse to resolve the field (not knowing which
          version to pick).  This parameter (if set to True) instead changes the error into
          assuming the caller wants the *first* variant.
        :param auto_map_final_newline_in_multiline_values: This parameter controls whether
          a multiline field with have / need a trailing newline. If True, the trailing
          newline is hidden on get and automatically added in set (if missing).
          When this option is set to True makes the dictionary behave more like the
          original Deb822 module.
        """
        return Deb822DictishParagraphWrapper(self, discard_comments_on_read=discard_comments_on_read, auto_map_initial_line_whitespace=auto_map_initial_line_whitespace, auto_resolve_ambiguous_fields=auto_resolve_ambiguous_fields, preserve_field_comments_on_field_updates=preserve_field_comments_on_field_updates, auto_map_final_newline_in_multiline_values=auto_map_final_newline_in_multiline_values)

    @property
    def _paragraph(self):
        return self

    def order_last(self, field):
        """Re-order the given field so it is "last" in the paragraph"""
        raise NotImplementedError

    def order_first(self, field):
        """Re-order the given field so it is "first" in the paragraph"""
        raise NotImplementedError

    def order_before(self, field, reference_field):
        """Re-order the given field so appears directly after the reference field in the paragraph

        The reference field must be present."""
        raise NotImplementedError

    def order_after(self, field, reference_field):
        """Re-order the given field so appears directly before the reference field in the paragraph

        The reference field must be present.
        """
        raise NotImplementedError

    @property
    def kvpair_count(self):
        raise NotImplementedError

    def iter_keys(self):
        raise NotImplementedError

    def contains_kvpair_element(self, item):
        raise NotImplementedError

    def get_kvpair_element(self, item, use_get=False):
        raise NotImplementedError

    def set_kvpair_element(self, key, value):
        raise NotImplementedError

    def remove_kvpair_element(self, key):
        raise NotImplementedError

    def sort_fields(self, key=None):
        """Re-order all fields

        :param key: Provide a key function (same semantics as for sorted).  Keep in mind that
          the module preserve the cases for field names - in generally, callers are recommended
          to use "lower()" to normalize the case.
        """
        raise NotImplementedError

    def set_field_to_simple_value(self, item, simple_value, *, preserve_original_field_comment=None, field_comment=None):
        """Sets a field in this paragraph to a simple "word" or "phrase"

        In many cases, it is better for callers to just use the paragraph as
        if it was a dictionary.  However, this method does enable to you choose
        the field comment (if any), which can be a reason for using it.

        This is suitable for "simple" fields like "Package".  Example:

            >>> example_deb822_paragraph = '''
            ... Package: foo
            ... '''
            >>> dfile = parse_deb822_file(example_deb822_paragraph.splitlines())
            >>> p = next(iter(dfile))
            >>> p.set_field_to_simple_value("Package", "mscgen")
            >>> p.set_field_to_simple_value("Architecture", "linux-any kfreebsd-any",
            ...                             field_comment=['Only ported to linux and kfreebsd'])
            >>> p.set_field_to_simple_value("Priority", "optional")
            >>> print(p.dump(), end='')
            Package: mscgen
            # Only ported to linux and kfreebsd
            Architecture: linux-any kfreebsd-any
            Priority: optional
            >>> # Values are formatted nicely by default, but it does not work with
            >>> # multi-line values
            >>> p.set_field_to_simple_value("Foo", "bar\\nbin\\n")
            Traceback (most recent call last):
                ...
            ValueError: Cannot use set_field_to_simple_value for values with newlines

        :param item: Name of the field to set.  If the paragraph already
          contains the field, then it will be replaced.  If the field exists,
          then it will preserve its order in the paragraph.  Otherwise, it is
          added to the end of the paragraph.
          Note this can be a "paragraph key", which enables you to control
          *which* instance of a field is being replaced (in case of duplicate
          fields).
        :param simple_value: The text to use as the value.  The value must not
          contain newlines.  Leading and trailing will be stripped but space
          within the value is preserved.  The value cannot contain comments
          (i.e. if the "#" token appears in the value, then it is considered
          a value rather than "start of a comment)
        :param preserve_original_field_comment: See the description for the
          parameter with the same name in the set_field_from_raw_string method.
        :param field_comment: See the description for the parameter with the same
          name in the set_field_from_raw_string method.
        """
        if '\n' in simple_value:
            raise ValueError('Cannot use set_field_to_simple_value for values with newlines')
        stripped = simple_value.strip()
        if stripped:
            raw_value = ' ' + stripped + '\n'
        else:
            raw_value = '\n'
        self.set_field_from_raw_string(item, raw_value, preserve_original_field_comment=preserve_original_field_comment, field_comment=field_comment)

    def set_field_from_raw_string(self, item, raw_string_value, *, preserve_original_field_comment=None, field_comment=None):
        """Sets a field in this paragraph to a given text value

        In many cases, it is better for callers to just use the paragraph as
        if it was a dictionary.  However, this method does enable to you choose
        the field comment (if any) and lets to have a higher degree of control
        over whitespace (on the first line), which can be a reason for using it.

        Example usage:

            >>> example_deb822_paragraph = '''
            ... Package: foo
            ... '''
            >>> dfile = parse_deb822_file(example_deb822_paragraph.splitlines())
            >>> p = next(iter(dfile))
            >>> raw_value = '''
            ... Build-Depends: debhelper-compat (= 12),
            ...                some-other-bd,
            ... # Comment
            ...                another-bd,
            ... '''.lstrip()  # Remove leading newline, but *not* the trailing newline
            >>> fname, new_value = raw_value.split(':', 1)
            >>> p.set_field_from_raw_string(fname, new_value)
            >>> print(p.dump(), end='')
            Package: foo
            Build-Depends: debhelper-compat (= 12),
                           some-other-bd,
            # Comment
                           another-bd,
            >>> # Format preserved

        :param item: Name of the field to set.  If the paragraph already
          contains the field, then it will be replaced.  Otherwise, it is
          added to the end of the paragraph.
          Note this can be a "paragraph key", which enables you to control
          *which* instance of a field is being replaced (in case of duplicate
          fields).
        :param raw_string_value: The text to use as the value.  The text must
          be valid deb822 syntax and is used *exactly* as it is given.
          Accordingly, multi-line values must include mandatory leading space
          on continuation lines, newlines after the value, etc. On the
          flip-side, any optional space or comments will be included.

          Note that the first line will *never* be read as a comment (if the
          first line of the value starts with a "#" then it will result
          in "Field-Name:#..." which is parsed as a value starting with "#"
          rather than a comment).
        :param preserve_original_field_comment: If True, then if there is an
          existing field and that has a comment, then the comment will remain
          after this operation.  This is the default is the `field_comment`
          parameter is omitted.
          Note that if the parameter is True and the item is ambiguous, this
          will raise an AmbiguousDeb822FieldKeyError.  When the parameter is
          omitted, the ambiguity is resolved automatically and if the resolved
          field has a comment then that will be preserved (assuming
          field_comment is None).
        :param field_comment: If not None, add or replace the comment for
          the field.  Each string in the list will become one comment
          line (inserted directly before the field name). Will appear in the
          same order as they do in the list.

          If you want complete control over the formatting of the comments,
          then ensure that each line start with "#" and end with "\\n" before
          the call.  Otherwise, leading/trailing whitespace is normalized
          and the missing "#"/"\\n" character is inserted.
        """
        new_content = []
        if preserve_original_field_comment is not None:
            if field_comment is not None:
                raise ValueError('The "preserve_original_field_comment" conflicts with "field_comment" parameter')
        elif field_comment is not None:
            if not isinstance(field_comment, Deb822CommentElement):
                new_content.extend((_format_comment(x) for x in field_comment))
                field_comment = None
            preserve_original_field_comment = False
        field_name, _, _ = _unpack_key(item)
        cased_field_name = field_name
        try:
            original = self.get_kvpair_element(item, use_get=True)
        except AmbiguousDeb822FieldKeyError:
            if preserve_original_field_comment:
                raise
            original = self.get_kvpair_element((field_name, 0), use_get=True)
        if preserve_original_field_comment is None:
            preserve_original_field_comment = True
        if original:
            cased_field_name = original.field_name
        raw = ':'.join((cased_field_name, raw_string_value))
        raw_lines = raw.splitlines(keepends=True)
        for i, line in enumerate(raw_lines, start=1):
            if not line.endswith('\n'):
                raise ValueError('Line {i} in new value was missing trailing newline'.format(i=i))
            if i != 1 and line[0] not in (' ', '\t', '#'):
                msg = 'Line {i} in new value was invalid.  It must either start with " " space (continuation line) or "#" (comment line). The line started with "{line}"'
                raise ValueError(msg.format(i=i, line=line[0]))
        if len(raw_lines) > 1 and raw_lines[-1].startswith('#'):
            raise ValueError('The last line in a value field cannot be a comment')
        new_content.extend(raw_lines)
        deb822_file = parse_deb822_file(iter(new_content))
        error_token = deb822_file.find_first_error_element()
        if error_token:
            raise ValueError('Syntax error in new field value for ' + field_name)
        paragraph = next(iter(deb822_file))
        assert isinstance(paragraph, Deb822NoDuplicateFieldsParagraphElement)
        value = paragraph.get_kvpair_element(field_name)
        assert value is not None
        if preserve_original_field_comment:
            if original:
                value.comment_element = original.comment_element
                original.comment_element = None
        elif field_comment is not None:
            value.comment_element = field_comment
        self.set_kvpair_element(item, value)

    @overload
    def dump(self, fd):
        pass

    @overload
    def dump(self):
        pass

    def dump(self, fd=None):
        if fd is None:
            return ''.join((t.text for t in self.iter_tokens()))
        for token in self.iter_tokens():
            fd.write(token.text.encode('utf-8'))
        return None