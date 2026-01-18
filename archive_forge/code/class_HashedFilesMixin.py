import json
import os
import posixpath
import re
from hashlib import md5
from urllib.parse import unquote, urldefrag, urlsplit, urlunsplit
from django.conf import STATICFILES_STORAGE_ALIAS, settings
from django.contrib.staticfiles.utils import check_settings, matches_patterns
from django.core.exceptions import ImproperlyConfigured
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage, storages
from django.utils.functional import LazyObject
class HashedFilesMixin:
    default_template = 'url("%(url)s")'
    max_post_process_passes = 5
    support_js_module_import_aggregation = False
    _js_module_import_aggregation_patterns = ('*.js', (('(?P<matched>import(?s:(?P<import>[\\s\\{].*?))\\s*from\\s*[\'"](?P<url>[./].*?)["\']\\s*;)', 'import%(import)s from "%(url)s";'), ('(?P<matched>export(?s:(?P<exports>[\\s\\{].*?))\\s*from\\s*["\'](?P<url>[./].*?)["\']\\s*;)', 'export%(exports)s from "%(url)s";'), ('(?P<matched>import\\s*[\'"](?P<url>[./].*?)["\']\\s*;)', 'import"%(url)s";'), ('(?P<matched>import\\(["\'](?P<url>.*?)["\']\\))', 'import("%(url)s")')))
    patterns = (('*.css', ('(?P<matched>url\\([\'"]{0,1}\\s*(?P<url>.*?)["\']{0,1}\\))', ('(?P<matched>@import\\s*["\']\\s*(?P<url>.*?)["\'])', '@import url("%(url)s")'), ('(?m)^(?P<matched>/\\*#[ \\t](?-i:sourceMappingURL)=(?P<url>.*)[ \\t]*\\*/)$', '/*# sourceMappingURL=%(url)s */'))), ('*.js', (('(?m)^(?P<matched>//# (?-i:sourceMappingURL)=(?P<url>.*))$', '//# sourceMappingURL=%(url)s'),)))
    keep_intermediate_files = True

    def __init__(self, *args, **kwargs):
        if self.support_js_module_import_aggregation:
            self.patterns += (self._js_module_import_aggregation_patterns,)
        super().__init__(*args, **kwargs)
        self._patterns = {}
        self.hashed_files = {}
        for extension, patterns in self.patterns:
            for pattern in patterns:
                if isinstance(pattern, (tuple, list)):
                    pattern, template = pattern
                else:
                    template = self.default_template
                compiled = re.compile(pattern, re.IGNORECASE)
                self._patterns.setdefault(extension, []).append((compiled, template))

    def file_hash(self, name, content=None):
        """
        Return a hash of the file with the given name and optional content.
        """
        if content is None:
            return None
        hasher = md5(usedforsecurity=False)
        for chunk in content.chunks():
            hasher.update(chunk)
        return hasher.hexdigest()[:12]

    def hashed_name(self, name, content=None, filename=None):
        parsed_name = urlsplit(unquote(name))
        clean_name = parsed_name.path.strip()
        filename = filename and urlsplit(unquote(filename)).path.strip() or clean_name
        opened = content is None
        if opened:
            if not self.exists(filename):
                raise ValueError("The file '%s' could not be found with %r." % (filename, self))
            try:
                content = self.open(filename)
            except OSError:
                return name
        try:
            file_hash = self.file_hash(clean_name, content)
        finally:
            if opened:
                content.close()
        path, filename = os.path.split(clean_name)
        root, ext = os.path.splitext(filename)
        file_hash = '.%s' % file_hash if file_hash else ''
        hashed_name = os.path.join(path, '%s%s%s' % (root, file_hash, ext))
        unparsed_name = list(parsed_name)
        unparsed_name[2] = hashed_name
        if '?#' in name and (not unparsed_name[3]):
            unparsed_name[2] += '?'
        return urlunsplit(unparsed_name)

    def _url(self, hashed_name_func, name, force=False, hashed_files=None):
        """
        Return the non-hashed URL in DEBUG mode.
        """
        if settings.DEBUG and (not force):
            hashed_name, fragment = (name, '')
        else:
            clean_name, fragment = urldefrag(name)
            if urlsplit(clean_name).path.endswith('/'):
                hashed_name = name
            else:
                args = (clean_name,)
                if hashed_files is not None:
                    args += (hashed_files,)
                hashed_name = hashed_name_func(*args)
        final_url = super().url(hashed_name)
        query_fragment = '?#' in name
        if fragment or query_fragment:
            urlparts = list(urlsplit(final_url))
            if fragment and (not urlparts[4]):
                urlparts[4] = fragment
            if query_fragment and (not urlparts[3]):
                urlparts[2] += '?'
            final_url = urlunsplit(urlparts)
        return unquote(final_url)

    def url(self, name, force=False):
        """
        Return the non-hashed URL in DEBUG mode.
        """
        return self._url(self.stored_name, name, force)

    def url_converter(self, name, hashed_files, template=None):
        """
        Return the custom URL converter for the given file name.
        """
        if template is None:
            template = self.default_template

        def converter(matchobj):
            """
            Convert the matched URL to a normalized and hashed URL.

            This requires figuring out which files the matched URL resolves
            to and calling the url() method of the storage.
            """
            matches = matchobj.groupdict()
            matched = matches['matched']
            url = matches['url']
            if re.match('^[a-z]+:', url):
                return matched
            if url.startswith('/') and (not url.startswith(settings.STATIC_URL)):
                return matched
            url_path, fragment = urldefrag(url)
            if not url_path:
                return matched
            if url_path.startswith('/'):
                assert url_path.startswith(settings.STATIC_URL)
                target_name = url_path.removeprefix(settings.STATIC_URL)
            else:
                source_name = name if os.sep == '/' else name.replace(os.sep, '/')
                target_name = posixpath.join(posixpath.dirname(source_name), url_path)
            hashed_url = self._url(self._stored_name, unquote(target_name), force=True, hashed_files=hashed_files)
            transformed_url = '/'.join(url_path.split('/')[:-1] + hashed_url.split('/')[-1:])
            if fragment:
                transformed_url += ('?#' if '?#' in url else '#') + fragment
            matches['url'] = unquote(transformed_url)
            return template % matches
        return converter

    def post_process(self, paths, dry_run=False, **options):
        """
        Post process the given dictionary of files (called from collectstatic).

        Processing is actually two separate operations:

        1. renaming files to include a hash of their content for cache-busting,
           and copying those files to the target storage.
        2. adjusting files which contain references to other files so they
           refer to the cache-busting filenames.

        If either of these are performed on a file, then that file is considered
        post-processed.
        """
        if dry_run:
            return
        hashed_files = {}
        adjustable_paths = [path for path in paths if matches_patterns(path, self._patterns)]
        processed_adjustable_paths = {}
        for name, hashed_name, processed, _ in self._post_process(paths, adjustable_paths, hashed_files):
            if name not in adjustable_paths or isinstance(processed, Exception):
                yield (name, hashed_name, processed)
            else:
                processed_adjustable_paths[name] = (name, hashed_name, processed)
        paths = {path: paths[path] for path in adjustable_paths}
        substitutions = False
        for i in range(self.max_post_process_passes):
            substitutions = False
            for name, hashed_name, processed, subst in self._post_process(paths, adjustable_paths, hashed_files):
                processed_adjustable_paths[name] = (name, hashed_name, processed)
                substitutions = substitutions or subst
            if not substitutions:
                break
        if substitutions:
            yield ('All', None, RuntimeError('Max post-process passes exceeded.'))
        self.hashed_files.update(hashed_files)
        yield from processed_adjustable_paths.values()

    def _post_process(self, paths, adjustable_paths, hashed_files):

        def path_level(name):
            return len(name.split(os.sep))
        for name in sorted(paths, key=path_level, reverse=True):
            substitutions = True
            storage, path = paths[name]
            with storage.open(path) as original_file:
                cleaned_name = self.clean_name(name)
                hash_key = self.hash_key(cleaned_name)
                if hash_key not in hashed_files:
                    hashed_name = self.hashed_name(name, original_file)
                else:
                    hashed_name = hashed_files[hash_key]
                if hasattr(original_file, 'seek'):
                    original_file.seek(0)
                hashed_file_exists = self.exists(hashed_name)
                processed = False
                if name in adjustable_paths:
                    old_hashed_name = hashed_name
                    try:
                        content = original_file.read().decode('utf-8')
                    except UnicodeDecodeError as exc:
                        yield (name, None, exc, False)
                    for extension, patterns in self._patterns.items():
                        if matches_patterns(path, (extension,)):
                            for pattern, template in patterns:
                                converter = self.url_converter(name, hashed_files, template)
                                try:
                                    content = pattern.sub(converter, content)
                                except ValueError as exc:
                                    yield (name, None, exc, False)
                    if hashed_file_exists:
                        self.delete(hashed_name)
                    content_file = ContentFile(content.encode())
                    if self.keep_intermediate_files:
                        self._save(hashed_name, content_file)
                    hashed_name = self.hashed_name(name, content_file)
                    if self.exists(hashed_name):
                        self.delete(hashed_name)
                    saved_name = self._save(hashed_name, content_file)
                    hashed_name = self.clean_name(saved_name)
                    if old_hashed_name == hashed_name:
                        substitutions = False
                    processed = True
                if not processed:
                    if not hashed_file_exists:
                        processed = True
                        saved_name = self._save(hashed_name, original_file)
                        hashed_name = self.clean_name(saved_name)
                hashed_files[hash_key] = hashed_name
                yield (name, hashed_name, processed, substitutions)

    def clean_name(self, name):
        return name.replace('\\', '/')

    def hash_key(self, name):
        return name

    def _stored_name(self, name, hashed_files):
        name = posixpath.normpath(name)
        cleaned_name = self.clean_name(name)
        hash_key = self.hash_key(cleaned_name)
        cache_name = hashed_files.get(hash_key)
        if cache_name is None:
            cache_name = self.clean_name(self.hashed_name(name))
        return cache_name

    def stored_name(self, name):
        cleaned_name = self.clean_name(name)
        hash_key = self.hash_key(cleaned_name)
        cache_name = self.hashed_files.get(hash_key)
        if cache_name:
            return cache_name
        intermediate_name = name
        for i in range(self.max_post_process_passes + 1):
            cache_name = self.clean_name(self.hashed_name(name, content=None, filename=intermediate_name))
            if intermediate_name == cache_name:
                self.hashed_files[hash_key] = cache_name
                return cache_name
            else:
                intermediate_name = cache_name
        raise ValueError("The name '%s' could not be hashed with %r." % (name, self))