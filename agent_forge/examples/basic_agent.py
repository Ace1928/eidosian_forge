from agent_forge.agent.agent import Agent
from agent_forge.core.memory import Memory
from agent_forge.core.model import get_model


def main():
    # Initialize model (assuming it's a wrapper around an LLM)
    model = get_model()

    # Create memory system
    memory = Memory()

    # Initialize agent
    agent = Agent(name="EidosTest", model=model, memory=memory)

    # Run a simple task
    response = agent.run(
        "Explain what Agent Forge is and tell a joke about AI frameworks"
    )
    print(f"Agent response: {response}")


if __name__ == "__main__":
    main()
