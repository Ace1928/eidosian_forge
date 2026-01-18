from lxml import etree

Limited XInclude support for the ElementTree package.

While lxml.etree has full support for XInclude (see
`etree.ElementTree.xinclude()`), this module provides a simpler, pure
Python, ElementTree compatible implementation that supports a simple
form of custom URL resolvers.
