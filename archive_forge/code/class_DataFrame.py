from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
class DataFrame(robjects.DataFrame):
    """DataFrame object extending the object of the same name in
    `rpy2.robjects.vectors` with functionalities defined in the R
    package `dplyr`."""

    @property
    def is_grouped_df(self) -> bool:
        """Is the DataFrame in a grouped state"""
        return dplyr.is_grouped_df(self)[0]

    def __rshift__(self, other):
        return other(self)

    @result_as
    def anti_join(self, *args, **kwargs):
        """Call the R function `dplyr::anti_join()`."""
        res = dplyr.anti_join(self, *args, **kwargs)
        return res

    @result_as
    def arrange(self, *args, _by_group=False):
        """Call the R function `dplyr::arrange()`."""
        res = dplyr.arrange(self, *args, **{'.by_group': _by_group})
        return res

    def copy_to(self, destination, name, **kwargs):
        """
        - destination: database
        - name: table name in the destination database
        """
        res = dplyr.copy_to(destination, self, name=name)
        return guess_wrap_type(res)(res)

    @result_as
    def collapse(self, *args, **kwargs):
        """
        Call the function `collapse` in the R package `dplyr`.
        """
        return dplyr.collapse(self, *args, **kwargs)

    @result_as
    def collect(self, *args, **kwargs):
        """Call the function `collect` in the R package `dplyr`."""
        return dplyr.collect(self, *args, **kwargs)

    @result_as
    def count(self, *args, **kwargs):
        """Call the function `count` in the R package `dplyr`."""
        return dplyr.count(self, *args, **kwargs)

    @result_as
    def distinct(self, *args, _keep_all=False):
        """Call the R function `dplyr::distinct()`."""
        res = dplyr.distinct(self, *args, **{'.keep_all': _keep_all})
        return res

    @result_as
    def filter(self, *args, _preserve=False):
        """Call the R function `dplyr::filter()`."""
        res = dplyr.filter(self, *args, **{'.preserve': _preserve})
        return res

    def group_by(self, *args, _add=False, _drop=robjects.rl('group_by_drop_default(.data)')):
        """Call the R function `dplyr::group_by()`."""
        res = dplyr.group_by(self, *args, **{'.add': _add, '.drop': _drop})
        return GroupedDataFrame(res)

    @result_as
    def inner_join(self, *args, **kwargs):
        """Call the R function `dplyr::inner_join()`."""
        res = dplyr.inner_join(self, *args, **kwargs)
        return res

    @result_as
    def left_join(self, *args, **kwargs):
        """Call the R function `dplyr::left_join()`."""
        res = dplyr.left_join(self, *args, **kwargs)
        return res

    @result_as
    def full_join(self, *args, **kwargs):
        """Call the R function `dplyr::full_join()`."""
        res = dplyr.full_join(self, *args, **kwargs)
        return res

    @result_as
    def mutate(self, **kwargs):
        """Call the R function `dplyr::mutate()`."""
        res = dplyr.mutate(self, **kwargs)
        return res

    @result_as
    def mutate_all(self, *args, **kwargs):
        """Call the R function `dplyr::mutate_all()`."""
        res = dplyr.mutate_all(self, *args, **kwargs)
        return res

    @result_as
    def mutate_at(self, *args, **kwargs):
        """Call the R function `dplyr::mutate_at()`."""
        res = dplyr.mutate_at(self, *args, **kwargs)
        return res

    @result_as
    def mutate_if(self, *args, **kwargs):
        """Call the R function `dplyr::mutate_if()`."""
        res = dplyr.mutate_if(self, *args, **kwargs)
        return res

    @result_as
    def right_join(self, *args, **kwargs):
        """Call the R function `dplyr::right_join()`."""
        res = dplyr.right_join(self, *args, **kwargs)
        return res

    @result_as
    def sample_frac(self, *args):
        """Call the R function `dplyr::sample_frac()`."""
        res = dplyr.sample_frac(self, *args)
        return res

    @result_as
    def sample_n(self, *args):
        """Call the R function `dplyr::sample_n()`."""
        res = dplyr.sample_n(self, *args)
        return res

    @result_as
    def select(self, *args):
        """Call the R function `dplyr::select()`."""
        res = dplyr.select(self, *args)
        return res

    @result_as
    def semi_join(self, *args, **kwargs):
        """Call the R function `dplyr::semi_join()`."""
        res = dplyr.semi_join(self, *args, **kwargs)
        return res

    @result_as
    def slice(self, *args, **kwargs):
        """Call the R function `dplyr::slice()`."""
        res = dplyr.slice(self, *args, **kwargs)
        return res

    @result_as
    def slice_head(self, *args, **kwargs):
        """Call the R function `dplyr::slice_head()`."""
        res = dplyr.slice_head(self, *args, **kwargs)
        return res

    @result_as
    def slice_min(self, *args, **kwargs):
        """Call the R function `dplyr::slice_min()`."""
        res = dplyr.slice_min(self, *args, **kwargs)
        return res

    @result_as
    def slice_max(self, *args, **kwargs):
        """Call the R function `dplyr::slice_max()`."""
        res = dplyr.slice_max(self, *args, **kwargs)
        return res

    @result_as
    def slice_sample(self, *args, **kwargs):
        """Call the R function `dplyr::slice_sample()`."""
        res = dplyr.slice_sample(self, *args, **kwargs)
        return res

    @result_as
    def slice_tail(self, *args, **kwargs):
        """Call the R function `dplyr::slice_tail()`."""
        res = dplyr.slice_tail(self, *args, **kwargs)
        return res

    def summarize(self, *args, **kwargs):
        """Call the R function `dplyr::summarize()`.

        This can return a GroupedDataFrame or a DataFrame.
        """
        res = dplyr.summarize(self, *args, **kwargs)
        return guess_wrap_type(res)(res)
    summarise = summarize

    def summarize_all(self, *args, **kwargs):
        """Call the R function `dplyr::summarize_all()`.

        This can return a GroupedDataFrame or a DataFrame.
        """
        res = dplyr.summarize_all(self, *args, **kwargs)
        return guess_wrap_type(res)(res)
    summarise_all = summarize_all

    def summarize_at(self, *args, **kwargs):
        """Call the R function `dplyr::summarize_at()`.

        This can return a GroupedDataFrame or a DataFrame.
        """
        res = dplyr.summarize_at(self, *args, **kwargs)
        return guess_wrap_type(res)(res)
    summarise_at = summarize_at

    def summarize_if(self, *args, **kwargs):
        """Call the R function `dplyr::summarize_if()`.

        This can return a GroupedDataFrame or a DataFrame.
        """
        res = dplyr.summarize_if(self, *args, **kwargs)
        return guess_wrap_type(res)(res)
    summarise_if = summarize_if

    @result_as
    def tally(self, *args, **kwargs):
        """Call the R function `dplyr::transmute()`."""
        res = dplyr.tally(self, *args, **kwargs)
        return res

    @result_as
    def transmute(self, *args, **kwargs):
        """Call the R function `dplyr::transmute()`."""
        res = dplyr.transmute(self, *args, **kwargs)
        return res

    @result_as
    def transmute_all(self, *args, **kwargs):
        """Call the R function `dplyr::transmute_all()`."""
        res = dplyr.transmute_all(self, *args, **kwargs)
        return res

    @result_as
    def transmute_at(self, *args, **kwargs):
        """Call the R function `dplyr::transmute_at()`."""
        res = dplyr.transmute_at(self, *args, **kwargs)
        return res

    @result_as
    def transmute_if(self, *args, **kwargs):
        """Call the R function `dplyr::transmute_if()`."""
        res = dplyr.transmute_if(self, *args, **kwargs)
        return res
    union = _wrap2(dplyr.union_data_frame, None)
    intersect = _wrap2(dplyr.intersect_data_frame, None)
    setdiff = _wrap2(dplyr.setdiff_data_frame, None)