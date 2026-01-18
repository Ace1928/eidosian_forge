from __future__ import annotations
import copy
from typing import TYPE_CHECKING, Any, Generic, Mapping, Optional, Type, Union
from bson import CodecOptions, _bson_to_dict
from bson.raw_bson import RawBSONDocument
from bson.timestamp import Timestamp
from pymongo import _csot, common
from pymongo.aggregation import (
from pymongo.collation import validate_collation_or_none
from pymongo.command_cursor import CommandCursor
from pymongo.errors import (
from pymongo.typings import _CollationIn, _DocumentType, _Pipeline
Advance the cursor without blocking indefinitely.

        This method returns the next change document without waiting
        indefinitely for the next change. For example::

            with db.collection.watch() as stream:
                while stream.alive:
                    change = stream.try_next()
                    # Note that the ChangeStream's resume token may be updated
                    # even when no changes are returned.
                    print("Current resume token: %r" % (stream.resume_token,))
                    if change is not None:
                        print("Change document: %r" % (change,))
                        continue
                    # We end up here when there are no recent changes.
                    # Sleep for a while before trying again to avoid flooding
                    # the server with getMore requests when no changes are
                    # available.
                    time.sleep(10)

        If no change document is cached locally then this method runs a single
        getMore command. If the getMore yields any documents, the next
        document is returned, otherwise, if the getMore returns no documents
        (because there have been no changes) then ``None`` is returned.

        :Returns:
          The next change document or ``None`` when no document is available
          after running a single getMore or when the cursor is closed.

        .. versionadded:: 3.8
        