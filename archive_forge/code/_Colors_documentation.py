from reportlab.lib import colors
Map float (red, green, blue) tuple to a ReportLab Color object.

        - values: A tuple of (red, green, blue) intensities as floats
          in the range 0 -> 1

        Takes a tuple of (red, green, blue) intensity values in the range
        0 -> 1 and returns an appropriate colors.Color object.
        