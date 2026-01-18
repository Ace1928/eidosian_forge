from parlai.mturk.tasks.acute_eval.run import AcuteEvaluator, add_args

Example script for running ACUTE-EVAL.
The only argument that *must* be modified for this to be run is:
``pairings_filepath``:  Path to pairings file in the format specified in the README.md

The following args are useful to tweak to fit your specific needs;
    - ``annotations_per_pair``: A useful arg if you'd like to evaluate a given conversation pair
                                more than once.
    - ``num_matchup_pairs``:    Essentially, how many pairs of conversations you would like to evaluate
    - ``subtasks_per_hit``:     How many comparisons you'd like a turker to complete in one HIT

Help strings for the other arguments can be found in run.py.
