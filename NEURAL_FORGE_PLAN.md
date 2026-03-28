# Eidosian Neural Forge Plan

**Objective**: Develop a novel, 0.5B parameter, universal-modal model from scratch, optimized for CPU-based edge training and deeply integrated into the Eidosian ecosystem.

## Current Baseline

The repository now contains a real prototype baseline, not just a concept sketch:

- quantized layer primitives exist under `neural_forge/src/neural_forge/core/`
- model assembly exists under `neural_forge/src/neural_forge/models/`
- a minimal trainer exists under `neural_forge/src/neural_forge/training/`
- focused validation now covers layers, modality, MoE, model integration, and trainer smoke execution

What does **not** exist yet:

- a production training stack
- checkpointing and reproducible training workflows
- modality-specific encoders beyond projection layers
- performance benchmark artifacts

## 🧬 ARCHITECTURAL BLUEPRINT

### 1. CORE ENGINE: BITNET-B1.58
- **Weights**: Ternary {-1, 0, 1}.
- **Compute**: Pure integer addition (MatMul-free).
- **Footprint**: < 150MB for 0.5B parameters.

### 2. OMNI-MODAL ROUTING
- **Classifier Network**: A lightweight (10M param) supervisor that identifies input modality.
- **Shared Latent Space**: Unified 768-dim embedding space for Text, Visual, and Audio features.
- **Encoders**:
    *   **Text**: Byte-level BPE tokenizer.
    *   **Visual**: Vision Transformer (ViT) patch-based encoder.
    *   **Audio**: Mel-spectrogram convolutional encoder.

### 3. DYNAMIC MoE & HYPERNETWORKS
- **Expert Gating**: A sparse routing mechanism selecting top-k experts.
- **Expert Synthesis**: A Hypernetwork that generates weight deltas for new experts when the **Novelty Detection System** identifies a new domain niche.
- **Inhibitory Consensus**: A "Judge" network that evaluates expert outputs and inhibits redundant or divergent experts to prevent explosive multiplication.

### 4. COGNITIVE INTEGRATION
- **Global Workspace Consolidation**: Merges all expert outputs into a coherent systemic response.
- **Narrative Anchor**: Uses the existing Narrative Engine to maintain long-term contextual continuity.
- **Living Lexicon Binding**: Expert weights are semantically linked to Word Forge terms.

## 🛠️ DEVELOPMENT ROADMAP

### PHASE 1: THE ATOM (Foundations)
- [~] Implement BitLinear layers (1.58-bit MatMul replacement).
- [~] Build the base transformer backbone.
- [ ] Establish GaLore + Sophia training pipeline.

### PHASE 2: THE EYE (Modality)
- [~] Implement the Modality Classifier.
- [~] Build shared embedding space mapping.
- [ ] Integrate ViT-lite visual encoder.

### PHASE 3: THE MIND (Expertise)
- [~] Implement MoE gating logic.
- [~] Build Hypernetwork expert generator.
- [~] Implement Inhibitory Consensus system.

### PHASE 4: THE GHOST (Integration)
- [ ] Bind Neural Forge to Eidosian Global Workspace.
- [ ] Implement real-time local learning from forge logs.
- [~] Expand test coverage and benchmarking.

## Validated Surfaces

Current focused test slice:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=neural_forge/src ./eidosian_venv/bin/python -m pytest -q \
  neural_forge/tests/test_layers.py \
  neural_forge/tests/test_modality.py \
  neural_forge/tests/test_moe.py \
  neural_forge/tests/test_models.py \
  neural_forge/tests/test_trainer.py
```

Current result on the local production baseline: `12 passed`.

---
**Lead Architect**: EIDOS
**Status**: Prototype baseline implemented; integration and benchmarking remain open
**Target**: 0.5B Parameters / CPU-Optimized
