#!/usr/bin/env bash
# Eidos E3 bootstrap: scaffold dirs, seed configs, prep venv. Idempotent.

set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

say() { printf "[bootstrap] %s\n" "$*"; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Missing: $1"; exit 1; }; }

usage() { echo "Usage: scripts/bootstrap.sh [--full]"; }

FULL=0
for arg in "$@"; do
  case "$arg" in
    --full) FULL=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

main() {
  need_cmd bash
  need_cmd python3

  say "Creating base directories"
  mkdir -p "$root_dir"/{cfg,core,scripts,state/{events,vector_store,weights,adapters,snaps},tests}

  if [ "$FULL" -eq 1 ]; then
    say "Creating extended directories (--full)"
    mkdir -p "$root_dir"/{planners/templates,actuators,instruments,reflect,ui,psyche,memory,learners,meta,embodiment,protocols/keys,net,sensors/camera/calib_db,sensors/{imu,lidar,audio,power,env},actuators_hw/{base,manip,io},control,safety,teleop,sim,datasets/sessions,ops,curriculum/{syllabus,generators,grader,bank},arenas,oracles/{property_tests,contracts,fuzzers},curiosity,forge/{dsl,catalog},model_lab/{adapters,distillers,calibration},atlas/{maps,roadmaps,chronicle},introspection,charter/audits,social,economy,population/genealogy,backups/parity,creative/{stylebook,moodboard},telemetry/dashboards,provenance/repro,legal,rituals,grants/{proposals,audits}}
  fi

  say "Seeding minimal config files (only if absent)"
  seed_cfg

  say "Making Python venv"
  if [ ! -d "$root_dir/.venv" ]; then
    python3 -m venv "$root_dir/.venv"
  fi
  # shellcheck disable=SC1091
  source "$root_dir/.venv/bin/activate"
  python -m pip install -U pip wheel >/dev/null
  python -m pip install --quiet 'pyyaml>=6,<7'

  say "Touch state files"
  : >"$root_dir/state/events/.keep"
  : >"$root_dir/state/vector_store/.keep"
  : >"$root_dir/state/weights/.keep"
  : >"$root_dir/state/adapters/.keep"
  : >"$root_dir/state/snaps/.keep"

  say "Summary: $(python -V 2>&1), PyYAML=$(python -c 'import yaml,sys;print(yaml.__version__)')"
  say "Done. Next: add core/state.py migrations and core/config.py loader."
}

seed_cfg() {
  if [ ! -d "$root_dir/cfg.sample" ]; then
    say "WARNING: cfg.sample missing; cannot seed cfg/"
    return 0
  fi
  mkdir -p "$root_dir/cfg"
  for f in self.yaml drives.yaml budgets.yaml policies.yaml skills.yaml; do
    if [ ! -f "$root_dir/cfg/$f" ]; then
      cp "$root_dir/cfg.sample/$f" "$root_dir/cfg/$f"
    fi
  done
}

main "$@"
