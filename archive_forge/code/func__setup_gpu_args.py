import argparse
import os
import uuid
from pathlib import Path
import submitit
from xformers.benchmarks.LRA.run_tasks import benchmark, get_arg_parser
def _setup_gpu_args(self):
    job_env = submitit.JobEnvironment()
    self.args.checkpoint_dir = Path(str(self.args.checkpoint_dir).replace('%j', str(job_env.job_id)))
    self.args.gpu = job_env.local_rank
    self.args.rank = job_env.global_rank
    self.args.world_size = job_env.num_tasks
    print(f'Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}')