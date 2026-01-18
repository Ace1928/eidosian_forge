import re
Parse the results from the other methods.

    The remaining methods are grouped together. Statistics
    for all three are listed for each of the pairwise
    species comparisons, with each method's results on its
    own line.
    The stats in this section must be handled differently
    due to the possible presence of NaN values, which won't
    get caught by my typical "line_floats" method used above.
    