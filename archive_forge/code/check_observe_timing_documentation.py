import textwrap
import timeit
 Prints a readable benchmark report.

    Parameters
    ----------
    descritption : str
        The description of the benchmarking scenario being reported.
    benchmark_template : str
        The format string used to print the times for the benchmark in a clean,
        formatted way
    get_time : function
        The function used to get the benchmark times for the current benchmark
        scenario
    get_time_args : list of tuples
        The list of tuples containing the arguments to be passed the the
        get_time function.  Note the first argument should give specifics about
        the case being timed and will be printed as part of the report.  e.g.
        ("@observe('name')"), or ("depends_on='name'", '@cached_property').
        The first item in the list will be used as a control.
    