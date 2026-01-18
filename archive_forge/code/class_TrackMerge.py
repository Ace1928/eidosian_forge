import os
from ..base import (
class TrackMerge(CommandLine):
    """
    Merges several TrackVis track files into a single track
    file.

    An id type property tag is added to each track in the
    newly merged file, with each unique id representing where
    the track was originally from. When the merged file is
    loaded in TrackVis, a property filter will show up in
    Track Property panel. Users can adjust that to distinguish
    and sub-group tracks by its id (origin).

    Example
    -------

    >>> import nipype.interfaces.diffusion_toolkit as dtk
    >>> mrg = dtk.TrackMerge()
    >>> mrg.inputs.track_files = ['track1.trk','track2.trk']
    >>> mrg.run()                                 # doctest: +SKIP
    """
    input_spec = TrackMergeInputSpec
    output_spec = TrackMergeOutputSpec
    _cmd = 'track_merge'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['track_file'] = os.path.abspath(self.inputs.output_file)
        return outputs