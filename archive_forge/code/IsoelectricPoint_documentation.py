Calculate and return the isoelectric point as float.

        This is a recursive function that uses bisection method.
        Wiki on bisection: https://en.wikipedia.org/wiki/Bisection_method

        Arguments:
         - pH: the pH at which the current charge of the protein is computed.
           This pH lies at the centre of the interval (mean of `min_` and `max_`).
         - min\_: the minimum of the interval. Initial value defaults to 4.05,
           which is below the theoretical minimum, when the protein is composed
           exclusively of aspartate.
         - max\_: the maximum of the the interval. Initial value defaults to 12,
           which is above the theoretical maximum, when the protein is composed
           exclusively of arginine.
        