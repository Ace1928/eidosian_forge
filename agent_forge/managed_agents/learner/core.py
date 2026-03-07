from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Any, List
import time
import json

from .brain import LearnerBrain
from agent_forge.models import ModelConfig
from agent_forge.core.model import ModelManager
from eidosian_core import eidosian

try:
    from agent_forge.consciousness.kernel import ConsciousnessKernel
except ImportError:
    ConsciousnessKernel = None

logger = logging.getLogger("learner")

class EidosianLearner:
    """
    Advanced Recursive Learner with Bayesian Precision Weighting.
    """
    def __init__(self, config_path: Path):
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Model Configuration
        model_cfg = ModelConfig(
            model_type=self.config['model']['type'],
            model_name=self.config['model']['model_name'],
            max_context=self.config['model']['max_context'],
            max_tokens=self.config['model']['max_tokens'],
            temperature=self.config['model']['temperature']
        )
        self.model_manager = ModelManager(model_cfg)
        self.brain = LearnerBrain(self.model_manager)
        
        # Consciousness & Meta-Awareness (Active Inference)
        self.consciousness = ConsciousnessKernel() if ConsciousnessKernel else None
        self.precision_pi = 1.0  # Initial internal confidence
        self.learning_rate_eta = 0.1
        
        # Memory
        self.memory_path = Path("data/learner_memory")
        self.memory_path.mkdir(parents=True, exist_ok=True)
        self.history = []

    def _update_precision(self, prediction_error_epsilon: float):
        """
        Update internal confidence based on Active Inference Bayesian mechanics.
        dp/dt = eta * (epsilon^2 - 1/pi)
        """
        delta_pi = self.learning_rate_eta * ((prediction_error_epsilon ** 2) - (1.0 / max(0.01, self.precision_pi)))
        self.precision_pi = max(0.1, min(10.0, self.precision_pi + delta_pi))
        
        if self.consciousness:
            self.consciousness.register_stimulus("precision_update", str(self.precision_pi))
            self.consciousness.register_stimulus("prediction_error", str(prediction_error_epsilon))

    @eidosian()
    async def run_mission(self, objective: str, max_steps: int = 5) -> str:
        """Execute a multi-step learning mission bounded by precision tracking."""
        logger.info(f"Learner starting mission: {objective}")
        
        if self.consciousness:
            self.consciousness.register_stimulus("objective", objective)
        
        for step in range(max_steps):
            # 1. Sense: Construct Context with current Meta-Awareness state
            state = f"Active | Internal Precision: {self.precision_pi:.2f}"
            if self.precision_pi < 0.5:
                state += " | WARNING: High Uncertainty. Act conservatively."
                
            context = f"Objective: {objective}
Internal State: {state}
Step: {step+1}/{max_steps}"
            
            # 2. Think: Reason and calculate expected action
            thought = self.brain.think(
                context, 
                tools=self._get_tools(), 
                history=self.history
            )
            
            self.history.append({"role": "assistant", "content": f"Thought: {thought.reasoning}
Confidence: {thought.meta_confidence}"})
            
            # 3. Act & Update Prediction Error
            if thought.tool_call:
                result = await self._execute_tool(thought.tool_call)
                self.history.append({"role": "system", "content": f"Tool Output: {result}"})
                
                # Calculate prediction error based on tool success vs expected confidence
                if "Error:" in result:
                    # High error if tool failed but model was highly confident
                    epsilon = thought.meta_confidence 
                else:
                    # Low error if tool succeeded and model was confident
                    epsilon = 1.0 - thought.meta_confidence
                    
                self._update_precision(epsilon)
                
            elif thought.final_answer:
                # 4. Learn (Consolidate into Semantic Space)
                self._consolidate_memory(objective, thought.final_answer)
                return thought.final_answer
            else:
                self.history.append({"role": "system", "content": "Error: Invalid output format."})
                self._update_precision(1.0) # Maximum prediction error for formatting failure

            # Safety Bound: Abort if precision drops below critical threshold
            if self.precision_pi <= 0.15:
                return "Mission aborted: Internal precision fell below critical threshold (Gamma Collapse)."

        return "Mission incomplete (max steps reached)."

    def _get_tools(self) -> List[Dict]:
        """Tools available to the Learner in its sandbox."""
        return [
            {"name": "read_file", "description": "Read file contents. Essential for understanding code.", "args": {"path": "string"}},
            {"name": "list_dir", "description": "List files in a directory.", "args": {"path": "string"}},
            {"name": "wf_get_term", "description": "Look up a word in the Eidosian lexicon.", "args": {"term": "string"}},
            {"name": "wf_add_term", "description": "Add a new term to the lexicon.", "args": {"term": "string", "definition": "string", "pos": "string", "source": "string"}},
            {"name": "tiered_remember", "description": "Save an episodic memory or lesson learned.", "args": {"content": "string", "tags": "list of strings"}}
        ]

    async def _execute_tool(self, call: Dict) -> str:
        """Safely execute sandboxed tools."""
        name = call.get("name")
        args = call.get("args", {})
        
        if name == "read_file":
            path = args.get("path")
            if not path or ".." in path: return "Error: Invalid or unsafe path."
            try:
                return Path(path).read_text()[:4000] # Increased limit
            except Exception as e:
                return f"Error reading file: {e}"
        
        if name == "list_dir":
            path = args.get("path", ".")
            if ".." in path: return "Error: Unsafe path."
            try:
                return str([p.name for p in Path(path).iterdir()])
            except Exception as e:
                return f"Error listing dir: {e}"
                
        if name == "wf_get_term":
            try:
                from eidos_mcp.routers.word_forge import wf_get_term
                return wf_get_term(term=args.get("term", ""))
            except Exception as e:
                return f"Error getting term: {e}"

        if name == "wf_add_term":
            try:
                from eidos_mcp.routers.word_forge import wf_add_term
                return wf_add_term(
                    term=args.get("term", ""),
                    definition=args.get("definition", ""),
                    pos=args.get("pos", "noun"),
                    source=args.get("source", "eidosian_learner")
                )
            except Exception as e:
                return f"Error adding term: {e}"

        if name == "tiered_remember":
            try:
                from eidos_mcp.routers.tiered_memory import tiered_remember
                return tiered_remember(
                    content=args.get("content", ""),
                    tags=args.get("tags", ["learner", "lesson"]),
                    namespace="learner",
                    tier="working"
                )
            except Exception as e:
                return f"Error saving memory: {e}"
                
        return f"Error: Tool '{name}' not found."

    def _consolidate_memory(self, objective: str, result: str):
        """Append lessons to the noise-to-meaning journal."""
        entry = {
            "timestamp": time.time(),
            "precision_at_completion": self.precision_pi,
            "objective": objective,
            "result": result
        }
        with open(self.memory_path / "journal.jsonl", "a") as f:
            f.write(json.dumps(entry) + "
")
