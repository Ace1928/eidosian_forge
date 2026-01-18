
An enumeration type that lists the curve tags supported by FreeType 2.

Each point of an outline has a specific tag which indicates whether it
describes a point used to control a line segment or an arc. The tags can take
the following values:

FT_CURVE_TAG_ON

  Used when the point is ‘on’ the curve. This corresponds to start and end
  points of segments and arcs. The other tags specify what is called an ‘off’
  point, i.e., a point which isn't located on the contour itself, but serves
  as a control point for a Bézier arc.

FT_CURVE_TAG_CONIC

  Used for an ‘off’ point used to control a conic Bézier arc.

FT_CURVE_TAG_CUBIC

  Used for an ‘off’ point used to control a cubic Bézier arc.


FT_Curve_Tag_On, FT_Curve_Tag_Conic, FT_Curve_Tag_Cubic are their
correspondning mixed-case aliases.

Use the FT_CURVE_TAG(tag) macro to filter out other, internally used flags.

