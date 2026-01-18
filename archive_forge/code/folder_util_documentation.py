from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
Reorders resources so containees are yielded before containers.

  For example, an iterator with the following:
  [
      gs://bucket/prefix/,
      gs://bucket/prefix/object,
      gs://bucket/prefix/object2,
      gs://bucket/prefix2/,
      gs://bucket/prefix2/object,
  ]

  Will become:
  [
      gs://bucket/prefix/object,
      gs://bucket/prefix/object2,
      gs://bucket/prefix/,
      gs://bucket/prefix2/object,
      gs://bucket/prefix2/,
  ]

  Args:
    ordered_iterator (Iterable[object]): Yields objects containing resources
      s.t. container resources are yielded before and contiguous with all of
      their containee resources. Bucket/folder/object resources ordered
      alphabetically by storage URL safisfy this constraint.
    get_url_function (None|object -> storage_url.StorageUrl): Maps objects
      yielded by `iterator` to storage URLs. Defaults to assuming yielded
      objects are URLs. Similar to the `key` attribute on the built-in
      list.sort() method.

  Yields:
    Resources s.t. containees are yielded before their containers, and
      contiguous with other containees.
  