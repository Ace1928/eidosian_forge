import gyp.common
import json
import os
import posixpath
def _AddSources(sources, base_path, base_path_components, result):
    """Extracts valid sources from |sources| and adds them to |result|. Each
  source file is relative to |base_path|, but may contain '..'. To make
  resolving '..' easier |base_path_components| contains each of the
  directories in |base_path|. Additionally each source may contain variables.
  Such sources are ignored as it is assumed dependencies on them are expressed
  and tracked in some other means."""
    for source in sources:
        if not len(source) or source.startswith('!!!') or source.startswith('$'):
            continue
        org_source = source
        source = source[0] + source[1:].replace('//', '/')
        if source.startswith('../'):
            source = _ResolveParent(source, base_path_components)
            if len(source):
                result.append(source)
            continue
        result.append(base_path + source)
        if debug:
            print('AddSource', org_source, result[len(result) - 1])