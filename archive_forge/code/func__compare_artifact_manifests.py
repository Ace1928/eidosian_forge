import filecmp
import logging
import os
import requests
import wandb
def _compare_artifact_manifests(src_art: wandb.Artifact, dst_art: wandb.Artifact) -> list:
    problems = []
    if isinstance(dst_art, wandb.CommError):
        return ['commError']
    if src_art.digest != dst_art.digest:
        problems.append(f'digest mismatch src_art.digest={src_art.digest!r}, dst_art.digest={dst_art.digest!r}')
    for name, src_entry in src_art.manifest.entries.items():
        dst_entry = dst_art.manifest.entries.get(name)
        if dst_entry is None:
            problems.append(f'missing manifest entry name={name!r}, src_entry={src_entry!r}')
            continue
        for attr in ['path', 'digest', 'size']:
            if getattr(src_entry, attr) != getattr(dst_entry, attr):
                problems.append(f'manifest entry mismatch attr={attr!r}, getattr(src_entry, attr)={getattr(src_entry, attr)!r}, getattr(dst_entry, attr)={getattr(dst_entry, attr)!r}')
    return problems