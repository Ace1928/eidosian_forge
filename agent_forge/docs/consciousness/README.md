# Consciousness Protocol Artifacts

This directory contains machine-readable RAC-AP protocol artifacts used by the `agent_forge` consciousness validator.

## Files
- `rac_ap_protocol_v1_2026_02_19.json`: default RAC-AP protocol schema and gating thresholds.

## CLI Usage
Validate built-in protocol:

```bash
./agent_forge/bin/eidctl consciousness protocol --json
```

Write a protocol template:

```bash
./agent_forge/bin/eidctl consciousness protocol \
  --write-template agent_forge/docs/consciousness/rac_ap_protocol_v1_2026_02_19.json
```

Run validation with explicit protocol:

```bash
./agent_forge/bin/eidctl consciousness validate \
  --dir state \
  --protocol agent_forge/docs/consciousness/rac_ap_protocol_v1_2026_02_19.json \
  --json
```

Create a pre-registration manifest:

```bash
./agent_forge/bin/eidctl consciousness preregister \
  --name "rac_ap_cycle" \
  --hypothesis "Winner-linked ignition improves intervention effects under perturbation." \
  --owner eidos \
  --json
```

## Research Anchors
- Global workspace architecture: https://pubmed.ncbi.nlm.nih.gov/17998071/
- GNW ignition framing: https://www.nature.com/articles/nrn3214
- Perturbational complexity index (PCI): https://www.science.org/doi/10.1126/scitranslmed.3006294
- Construct validity program: https://psycnet.apa.org/record/1956-03741-001
- Prompt-injection security baseline: https://www.ncsc.gov.uk/collection/ai-guidelines/guidelines-for-secure-ai-system-development/prompt-injection
- Interactive external benchmarks:
  - AgentBench: https://openreview.net/forum?id=zAdUB0aCTQ
  - WebArena: https://webarena.dev/
  - OSWorld: https://os-world.github.io/
