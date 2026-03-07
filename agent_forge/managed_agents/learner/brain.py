from __future__ import annotations
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from agent_forge.core.model import ModelManager
from eidosian_core import eidosian

@dataclass
class Thought:
    reasoning: str
    tool_call: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None
    meta_confidence: float = 1.0  # Added for precision weighting

class LearnerBrain:
    """The cognitive engine of the Eidosian Learner."""
    
    def __init__(self, model_manager: ModelManager):
        self.model = model_manager

    @eidosian()
    def think(self, context: str, tools: List[Dict], history: List[Dict]) -> Thought:
        """
        Execute a cognitive cycle: Observe Context -> Reason -> Decide Action.
        Uses a ReAct-style prompt adapted for small local models (Qwen).
        """
        tool_desc = json.dumps(tools, indent=2)
        history_text = "
".join([f"{h['role']}: {h['content']}" for h in history[-10:]])
        
        prompt = f"""<|im_start|>system
You are the Eidosian Learner, a highly advanced recursive self-improvement agent.
Your goal is to deeply analyze the codebase, identify complex patterns, use the lexicon, and propose comprehensive optimizations.
Take your time to reason thoroughly. You must output a structured response including a confidence score (0.0 to 1.0).

You have access to the following tools:
{tool_desc}

FORMAT INSTRUCTIONS:
To use a tool, you MUST use this format:
Thought: <your detailed reasoning>
Confidence: <0.0-1.0>
Action: <tool_name>
Action Input: <json_arguments>

To finish, use:
Thought: <your detailed reasoning and synthesis>
Confidence: <0.0-1.0>
Final Answer: <your comprehensive conclusion>

Example:
Thought: I need to check the file content to understand its structure, and then define a new term in the lexicon.
Confidence: 0.95
Action: read_file
Action Input: {{"path": "README.md"}}
<|im_end|>
<|im_start|>user
Current Context: {context}

Conversation History:
{history_text}

What is your next step? Take a deep breath and reason step-by-step.
<|im_end|>
<|im_start|>assistant
"""
        response = self.model.generate(prompt, max_tokens=2048, temperature=0.7)
        return self._parse_response(response)

    def _parse_response(self, text: str) -> Thought:
        """Parse the raw LLM output into a structured Thought with Confidence."""
        # Strip <think> blocks if present
        text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        
        thought_match = re.search(r"Thought:\s*(.*?)(?=(Confidence:|Action:|Final Answer:|$))", text, re.DOTALL)
        reasoning = thought_match.group(1).strip() if thought_match else "Proceeding..."
        
        confidence_match = re.search(r"Confidence:\s*([0-9.]+)", text)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        
        action_match = re.search(r"Action:\s*(\w+)", text)
        if action_match:
            tool_name = action_match.group(1)
            input_match = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)
            tool_args = {}
            if input_match:
                try:
                    tool_args = json.loads(input_match.group(1))
                except json.JSONDecodeError:
                    pass # Fail gracefully
            return Thought(reasoning=reasoning, tool_call={"name": tool_name, "args": tool_args}, meta_confidence=confidence)
        
        final_match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL)
        if final_match:
            return Thought(reasoning=reasoning, final_answer=final_match.group(1).strip(), meta_confidence=confidence)
            
        return Thought(reasoning=text, final_answer=None, meta_confidence=0.1) # High uncertainty fallback
