from fontTools.misc.textTools import bytesjoin, safeEval, strjoin, tobytes, tostr
from fontTools.misc import sstruct
from . import DefaultTable
from collections.abc import Sequence
from dataclasses import dataclass, astuple
from io import BytesIO
import struct
import logging
Compiles/decompiles SVG table.

https://docs.microsoft.com/en-us/typography/opentype/spec/svg

The XML format is:

.. code-block:: xml

	<SVG>
		<svgDoc endGlyphID="1" startGlyphID="1">
			<![CDATA[ <complete SVG doc> ]]
		</svgDoc>
	...
		<svgDoc endGlyphID="n" startGlyphID="m">
			<![CDATA[ <complete SVG doc> ]]
		</svgDoc>
	</SVG>
