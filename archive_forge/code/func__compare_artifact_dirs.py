import filecmp
import logging
import os
import requests
import wandb
def _compare_artifact_dirs(src_dir, dst_dir) -> list:

    def compare(src_dir, dst_dir):
        comparison = filecmp.dircmp(src_dir, dst_dir)
        differences = {'left_only': comparison.left_only, 'right_only': comparison.right_only, 'diff_files': comparison.diff_files, 'subdir_differences': {}}
        for subdir in comparison.subdirs:
            subdir_src = os.path.join(src_dir, subdir)
            subdir_dst = os.path.join(dst_dir, subdir)
            subdir_differences = compare(subdir_src, subdir_dst)
            if subdir_differences and any(subdir_differences.values()):
                differences['subdir_differences'][subdir] = subdir_differences
        if all((not diff for diff in differences.values())):
            return None
        return differences
    return compare(src_dir, dst_dir)